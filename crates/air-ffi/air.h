#ifndef AIR_H
#define AIR_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AirEngineHandle AirEngineHandle;

AirEngineHandle* air_engine_init(const char* model_path, const char* config_json);
void air_engine_free(AirEngineHandle* handle);
int32_t air_engine_generate(AirEngineHandle* handle, const char* prompt, char* output_buf, size_t output_buf_len);
int32_t air_server_start(uint16_t port);

#ifdef __cplusplus
}
#endif

#endif // AIR_H
