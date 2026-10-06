/*
 * Lora - firmware de la placa de audio (Waveshare ESP32-S3-AUDIO-Board).
 *
 * Convierte la placa en una tarjeta de sonido USB (UAC 2.0) para la Raspberry
 * Pi: los microfonos entran al host como dispositivo de captura y el altavoz
 * como dispositivo de reproduccion. Toda la inteligencia (STT, LLM, TTS) vive
 * en la Pi; esta placa solo capta y reproduce.
 *
 * Pines segun el diagrama oficial de Waveshare -- ver README.md.
 */

#include <string.h>

#include "driver/i2c_master.h"
#include "driver/i2s_std.h"   /* define I2S_NUM_0 y los tipos del bus I2S */
#include "esp_check.h"
#include "esp_codec_dev.h"
#include "esp_codec_dev_defaults.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "usb_device_uac.h"

static const char *TAG = "lora_audio";

/* ---------------------------------------------------------------- pines --- */

#define I2C_PUERTO      I2C_NUM_0
#define PIN_I2C_SDA     11
#define PIN_I2C_SCL     10

#define I2S_PUERTO      I2S_NUM_0
#define PIN_I2S_MCLK    12
#define PIN_I2S_BCLK    13
#define PIN_I2S_WS      14
#define PIN_I2S_DOUT    16   /* ESP32 -> ES8311 (altavoz). La comunidad lo confirma: NO es 15 */
#define PIN_I2S_DIN     15   /* ES7210 -> ESP32 (microfonos). La comunidad lo confirma: NO es 16 */

/* El amplificador NS4150B NO cuelga de un GPIO del chip: esta en el expansor
 * TCA9555, bit 0 del puerto 1 (lo que Waveshare llama EXIO8). Si no se activa,
 * el altavoz queda mudo aunque el ES8311 este perfectamente configurado. */
#define TCA9555_ADDR        0x20
#define TCA9555_REG_SALIDA1 0x03
#define TCA9555_REG_CONFIG1 0x07
#define TCA9555_BIT_PA_EN   0x01

/* OJO con el convenio de direcciones: esp_codec_dev espera la direccion en
 * formato de 8 BITS, porque su capa I2C la desplaza ella misma
 * (audio_codec_ctrl_i2c.c: ".device_address = (i2c_cfg->addr >> 1)").
 *
 * El diagrama de Waveshare da las direcciones en 7 bits (ES8311 0x18,
 * ES7210 0x40), asi que hay que duplicarlas. Pasar el valor de 7 bits tal
 * cual hace que el componente hable con 0x0C y el codec responda NACK.
 *
 * El TCA9555 de mas arriba NO sigue este convenio: a ese lo manejamos con la
 * API cruda de IDF, que usa 7 bits sin desplazar. De ahi que 0x20 funcione. */
#define ES8311_ADDR     0x30   /* 7 bits: 0x18 */
#define ES7210_ADDR     0x80   /* 7 bits: 0x40 */

/* El host manda y pide audio a 16 kHz mono: es exactamente lo que consume el
 * STT de la Pi (whisper espera 16 kHz). Convertir en la placa evita que la Pi
 * gaste CPU remuestreando, que es justo el recurso que compite con Ollama. */
#define MUESTREO_HZ     16000
#define BITS_MUESTRA    16
#define CANALES_MIC     1
#define CANALES_SPK     1

static esp_codec_dev_handle_t s_altavoz;
static esp_codec_dev_handle_t s_microfono;
static i2c_master_bus_handle_t s_bus_i2c;
static i2s_chan_handle_t s_i2s_tx;
static i2s_chan_handle_t s_i2s_rx;

/* ----------------------------------------------------------------- I2S ---- */

/* En IDF 5.x el bus I2S NO lo crea esp_codec_dev: hay que instalarlo aparte y
 * pasarle los manejadores (lo dice audio_codec_i2s_cfg_t: "need provide on
 * IDF 5.x"). Si se le pasa NULL el firmware COMPILA pero el codec se queda sin
 * camino de datos y no se capta ni se reproduce nada.
 *
 * Los dos codecs cuelgan del mismo puerto en full duplex y comparten MCLK,
 * BCLK y WS: el ESP32 es el maestro y ambos codecs van como esclavos, que es
 * como los configuraba el firmware de fabrica. */
static esp_err_t iniciar_i2s(void)
{
    i2s_chan_config_t chan = I2S_CHANNEL_DEFAULT_CONFIG(I2S_PUERTO, I2S_ROLE_MASTER);
    chan.auto_clear = true;   /* que mande silencio en vez de repetir el ultimo buffer */
    ESP_RETURN_ON_ERROR(i2s_new_channel(&chan, &s_i2s_tx, &s_i2s_rx),
                        TAG, "no se pudieron crear los canales I2S");

    i2s_std_config_t std = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(MUESTREO_HZ),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = PIN_I2S_MCLK,
            .bclk = PIN_I2S_BCLK,
            .ws = PIN_I2S_WS,
            .dout = PIN_I2S_DOUT,
            .din = PIN_I2S_DIN,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };
    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(s_i2s_tx, &std), TAG, "I2S tx");
    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(s_i2s_rx, &std), TAG, "I2S rx");
    ESP_RETURN_ON_ERROR(i2s_channel_enable(s_i2s_tx), TAG, "habilitar tx");
    ESP_RETURN_ON_ERROR(i2s_channel_enable(s_i2s_rx), TAG, "habilitar rx");

    ESP_LOGI(TAG, "I2S listo: mclk=%d bclk=%d ws=%d dout=%d din=%d",
             PIN_I2S_MCLK, PIN_I2S_BCLK, PIN_I2S_WS, PIN_I2S_DOUT, PIN_I2S_DIN);
    return ESP_OK;
}

/* ------------------------------------------------------- amplificador ----- */

static esp_err_t habilitar_amplificador(void)
{
    i2c_master_dev_handle_t dev;
    i2c_device_config_t cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = TCA9555_ADDR,
        .scl_speed_hz = 100000,
    };
    ESP_RETURN_ON_ERROR(i2c_master_bus_add_device(s_bus_i2c, &cfg, &dev),
                        TAG, "no se pudo hablar con el TCA9555");

    /* Puerto 1 bit 0 como SALIDA (un 0 en el registro de configuracion). */
    uint8_t conf[2] = {TCA9555_REG_CONFIG1, (uint8_t)~TCA9555_BIT_PA_EN};
    ESP_RETURN_ON_ERROR(i2c_master_transmit(dev, conf, sizeof(conf), 100),
                        TAG, "fallo al configurar el expansor");

    /* Y ahora a 1 para encender el amplificador. */
    uint8_t alto[2] = {TCA9555_REG_SALIDA1, TCA9555_BIT_PA_EN};
    ESP_RETURN_ON_ERROR(i2c_master_transmit(dev, alto, sizeof(alto), 100),
                        TAG, "fallo al encender el amplificador");

    ESP_LOGI(TAG, "amplificador habilitado (EXIO8)");
    return ESP_OK;
}

/* ------------------------------------------------------------- codecs ----- */

static esp_err_t iniciar_audio(void)
{
    i2c_master_bus_config_t bus_cfg = {
        .i2c_port = I2C_PUERTO,
        .sda_io_num = PIN_I2C_SDA,
        .scl_io_num = PIN_I2C_SCL,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    ESP_RETURN_ON_ERROR(i2c_new_master_bus(&bus_cfg, &s_bus_i2c), TAG, "I2C");

    ESP_RETURN_ON_ERROR(habilitar_amplificador(), TAG, "amplificador");
    ESP_RETURN_ON_ERROR(iniciar_i2s(), TAG, "I2S");

    audio_codec_i2s_cfg_t i2s_cfg = {
        .port = I2S_PUERTO,
        .rx_handle = s_i2s_rx,
        .tx_handle = s_i2s_tx,
    };
    const audio_codec_data_if_t *datos = audio_codec_new_i2s_data(&i2s_cfg);
    ESP_RETURN_ON_FALSE(datos, ESP_FAIL, TAG, "no se pudo abrir el I2S");

    /* bus_handle es OBLIGATORIO desde IDF 5.3: sin el, audio_codec_new_i2c_ctrl
     * devuelve NULL en silencio y el codec falla despues con un "Wrong codec
     * config" que hace pensar que el chip no responde, cuando en realidad la
     * interfaz I2C nunca llego a crearse. Mismo patron que rx_handle/tx_handle
     * en el I2S: en IDF 5.x hay que pasar los manejadores, no solo el puerto. */
    audio_codec_i2c_cfg_t i2c_8311 = {
        .port = I2C_PUERTO,
        .addr = ES8311_ADDR,
        .bus_handle = s_bus_i2c,
    };
    const audio_codec_ctrl_if_t *ctrl_8311 = audio_codec_new_i2c_ctrl(&i2c_8311);
    ESP_RETURN_ON_FALSE(ctrl_8311, ESP_FAIL, TAG,
                        "no se pudo crear la interfaz I2C del ES8311");

    audio_codec_i2c_cfg_t i2c_7210 = {
        .port = I2C_PUERTO,
        .addr = ES7210_ADDR,
        .bus_handle = s_bus_i2c,
    };
    const audio_codec_ctrl_if_t *ctrl_7210 = audio_codec_new_i2c_ctrl(&i2c_7210);
    ESP_RETURN_ON_FALSE(ctrl_7210, ESP_FAIL, TAG,
                        "no se pudo crear la interfaz I2C del ES7210");

    es8311_codec_cfg_t cfg_8311 = {
        .ctrl_if = ctrl_8311,
        .codec_mode = ESP_CODEC_DEV_WORK_MODE_DAC,
        .use_mclk = true,
    };
    const audio_codec_if_t *if_8311 = es8311_codec_new(&cfg_8311);
    ESP_RETURN_ON_FALSE(if_8311, ESP_FAIL, TAG, "ES8311 no responde");

    es7210_codec_cfg_t cfg_7210 = {
        .ctrl_if = ctrl_7210,
        .mic_selected = ES7120_SEL_MIC1 | ES7120_SEL_MIC2,
    };
    const audio_codec_if_t *if_7210 = es7210_codec_new(&cfg_7210);
    ESP_RETURN_ON_FALSE(if_7210, ESP_FAIL, TAG, "ES7210 no responde");

    esp_codec_dev_cfg_t dev_spk = {
        .dev_type = ESP_CODEC_DEV_TYPE_OUT,
        .codec_if = if_8311,
        .data_if = datos,
    };
    s_altavoz = esp_codec_dev_new(&dev_spk);

    esp_codec_dev_cfg_t dev_mic = {
        .dev_type = ESP_CODEC_DEV_TYPE_IN,
        .codec_if = if_7210,
        .data_if = datos,
    };
    s_microfono = esp_codec_dev_new(&dev_mic);

    esp_codec_dev_sample_info_t fs = {
        .bits_per_sample = BITS_MUESTRA,
        .channel = 1,
        .sample_rate = MUESTREO_HZ,
    };
    ESP_RETURN_ON_ERROR(esp_codec_dev_open(s_altavoz, &fs), TAG, "abrir altavoz");
    ESP_RETURN_ON_ERROR(esp_codec_dev_open(s_microfono, &fs), TAG, "abrir mic");

    esp_codec_dev_set_out_vol(s_altavoz, 70);
    esp_codec_dev_set_in_gain(s_microfono, 30.0);

    ESP_LOGI(TAG, "codecs listos: %d Hz, %d bits, mono", MUESTREO_HZ, BITS_MUESTRA);
    return ESP_OK;
}

/* -------------------------------------------------------------- UAC ------- */

/* El host PIDE audio del microfono. Bloquea hasta tener datos: es lo que
 * recomienda el componente para no desincronizar la linea temporal del USB. */
static esp_err_t cb_microfono(uint8_t *buf, size_t largo, size_t *leidos, void *arg)
{
    (void)arg;
    if (esp_codec_dev_read(s_microfono, buf, largo) != ESP_CODEC_DEV_OK) {
        memset(buf, 0, largo);   /* silencio antes que ruido o un corte */
    }
    *leidos = largo;
    return ESP_OK;
}

/* El host ENVIA audio para el altavoz (la voz de Piper, la musica de Trivia). */
static esp_err_t cb_altavoz(uint8_t *buf, size_t largo, void *arg)
{
    (void)arg;
    esp_codec_dev_write(s_altavoz, buf, largo);
    return ESP_OK;
}

/* Ojo: el componente espera que estos dos devuelvan void, no esp_err_t. Si se
 * declaran devolviendo esp_err_t, el compilador rechaza la asignacion en
 * uac_device_config_t por tipo de puntero incompatible. */
static void cb_volumen(uint32_t volumen, void *arg)
{
    (void)arg;
    esp_codec_dev_set_out_vol(s_altavoz, (int)volumen);
}

static void cb_silencio(uint32_t silenciar, void *arg)
{
    (void)arg;
    esp_codec_dev_set_out_mute(s_altavoz, silenciar != 0);
}

/* ------------------------------------------------------- diagnostico ----- */
/*
 * Compilar con:
 *   idf.py -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.diag" \
 *          -DLORA_MODO_DIAG=1 build
 *
 * En este modo NO se inicializa el USB de audio, asi que el puerto serie
 * sobrevive y se pueden leer los resultados. Prueba las cuatro combinaciones
 * de las dos hipotesis que explican que el microfono entregue ceros exactos:
 * pines DIN/DOUT intercambiados, y ES7210 en TDM de 4 canales (que es como lo
 * configuraba el firmware de fabrica) en vez de I2S estandar.
 */
#ifdef LORA_MODO_DIAG

#include <math.h>   /* sqrt() para el nivel de senal */

static double medir_rms(int din, int dout, uint8_t mics, bool tdm)
{
    /* Se reconstruye todo en cada intento: los codecs guardan estado y
     * reutilizarlos entre configuraciones daria resultados enganosos. */
    i2s_chan_handle_t tx = NULL, rx = NULL;
    i2s_chan_config_t chan = I2S_CHANNEL_DEFAULT_CONFIG(I2S_PUERTO, I2S_ROLE_MASTER);
    chan.auto_clear = true;
    if (i2s_new_channel(&chan, &tx, &rx) != ESP_OK) {
        return -1.0;
    }

    i2s_std_config_t std = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(MUESTREO_HZ),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = PIN_I2S_MCLK,
            .bclk = PIN_I2S_BCLK,
            .ws = PIN_I2S_WS,
            .dout = dout,
            .din = din,
            .invert_flags = {false, false, false},
        },
    };
    if (i2s_channel_init_std_mode(tx, &std) != ESP_OK ||
        i2s_channel_init_std_mode(rx, &std) != ESP_OK ||
        i2s_channel_enable(tx) != ESP_OK || i2s_channel_enable(rx) != ESP_OK) {
        i2s_del_channel(tx);
        i2s_del_channel(rx);
        return -2.0;
    }

    audio_codec_i2s_cfg_t i2s_cfg = {.port = I2S_PUERTO, .rx_handle = rx, .tx_handle = tx};
    const audio_codec_data_if_t *datos = audio_codec_new_i2s_data(&i2s_cfg);

    audio_codec_i2c_cfg_t i2c_7210 = {
        .port = I2C_PUERTO, .addr = ES7210_ADDR, .bus_handle = s_bus_i2c};
    const audio_codec_ctrl_if_t *ctrl = audio_codec_new_i2c_ctrl(&i2c_7210);

    es7210_codec_cfg_t cfg = {
        .ctrl_if = ctrl,
        .mic_selected = mics,
        .master_mode = false,
    };
    const audio_codec_if_t *codec = es7210_codec_new(&cfg);
    if (codec == NULL) {
        i2s_del_channel(tx);
        i2s_del_channel(rx);
        return -3.0;
    }

    esp_codec_dev_cfg_t dev_cfg = {
        .dev_type = ESP_CODEC_DEV_TYPE_IN, .codec_if = codec, .data_if = datos};
    esp_codec_dev_handle_t dev = esp_codec_dev_new(&dev_cfg);

    esp_codec_dev_sample_info_t fs = {
        .bits_per_sample = BITS_MUESTRA,
        .channel = tdm ? 4 : 1,
        .sample_rate = MUESTREO_HZ,
    };
    double rms = -4.0;
    if (esp_codec_dev_open(dev, &fs) == ESP_CODEC_DEV_OK) {
        esp_codec_dev_set_in_gain(dev, 30.0);
        vTaskDelay(pdMS_TO_TICKS(300));   /* que asiente el codec */

        static int16_t buf[4096];
        double suma = 0.0;
        int total = 0;
        for (int i = 0; i < 8; i++) {
            if (esp_codec_dev_read(dev, buf, sizeof(buf)) != ESP_CODEC_DEV_OK) {
                break;
            }
            for (size_t j = 0; j < sizeof(buf) / sizeof(buf[0]); j++) {
                suma += (double)buf[j] * (double)buf[j];
                total++;
            }
        }
        rms = total ? sqrt(suma / total) : -5.0;
        esp_codec_dev_close(dev);
    }

    esp_codec_dev_delete(dev);
    i2s_channel_disable(tx);
    i2s_channel_disable(rx);
    i2s_del_channel(tx);
    i2s_del_channel(rx);
    return rms;
}

static void diagnostico(void)
{
    ESP_LOGW(TAG, "===== DIAGNOSTICO DEL MICROFONO =====");
    ESP_LOGW(TAG, "Habla fuerte cerca de la placa durante toda la prueba.");
    ESP_LOGW(TAG, "RMS cercano a 0 = no capta. RMS > 50 = capta de verdad.");

    const uint8_t dos = ES7120_SEL_MIC1 | ES7120_SEL_MIC2;
    const uint8_t cuatro = ES7120_SEL_MIC1 | ES7120_SEL_MIC2 |
                           ES7120_SEL_MIC3 | ES7120_SEL_MIC4;

    struct { int din; int dout; uint8_t mics; bool tdm; const char *nombre; } pruebas[] = {
        {16, 15, dos,    false, "A: din=16 dout=15, 2 mics, estandar (documentado)"},
        {15, 16, dos,    false, "B: din=15 dout=16, 2 mics, estandar (pines al reves)"},
        {16, 15, cuatro, true,  "C: din=16 dout=15, 4 mics, TDM (como el de fabrica)"},
        {15, 16, cuatro, true,  "D: din=15 dout=16, 4 mics, TDM"},
    };

    for (size_t i = 0; i < sizeof(pruebas) / sizeof(pruebas[0]); i++) {
        double rms = medir_rms(pruebas[i].din, pruebas[i].dout,
                               pruebas[i].mics, pruebas[i].tdm);
        if (rms < 0) {
            ESP_LOGE(TAG, "%s -> fallo al configurar (codigo %.0f)",
                     pruebas[i].nombre, rms);
        } else {
            ESP_LOGW(TAG, "%s -> RMS %.1f  %s", pruebas[i].nombre, rms,
                     rms > 50.0 ? "<<<<< CAPTA" : "(silencio)");
        }
        vTaskDelay(pdMS_TO_TICKS(500));
    }

    ESP_LOGW(TAG, "===== FIN DEL DIAGNOSTICO =====");
}

#endif /* LORA_MODO_DIAG */

void app_main(void)
{
    ESP_LOGI(TAG, "Lora - placa de audio como tarjeta de sonido USB");

#ifdef LORA_MODO_DIAG
    /* En diagnostico se necesita el I2C (para hablar con los codecs y el
     * expansor) pero NO se toca el USB, para no perder el puerto serie. */
    i2c_master_bus_config_t bus_cfg = {
        .i2c_port = I2C_PUERTO,
        .sda_io_num = PIN_I2C_SDA,
        .scl_io_num = PIN_I2C_SCL,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_cfg, &s_bus_i2c));
    habilitar_amplificador();
    diagnostico();
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
#else

    if (iniciar_audio() != ESP_OK) {
        /* Falla ruidosa a proposito: una placa que enumera como tarjeta de
         * sonido pero no capta ni reproduce es peor que una que no enumera,
         * porque el fallo se descubre mucho mas tarde. */
        ESP_LOGE(TAG, "no se pudo iniciar el audio; no se expone el USB");
        return;
    }

    uac_device_config_t uac = {
        .output_cb = cb_altavoz,
        .input_cb = cb_microfono,
        .set_mute_cb = cb_silencio,
        .set_volume_cb = cb_volumen,
        .cb_ctx = NULL,
    };
    ESP_ERROR_CHECK(uac_device_init(&uac));

    ESP_LOGI(TAG, "USB listo: la Pi deberia verla con 'arecord -l' y 'aplay -l'");

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
#endif /* LORA_MODO_DIAG */
}
