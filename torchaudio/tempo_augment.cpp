#include <iostream>
#include <sox.h>
#include <assert.h>
#include <cstring>

using namespace std;


struct ds_audio_buffer {
  char*  buffer;
  size_t buffer_size;
  int    sample_rate;
};


int main() {
    sox_init();
    
    struct ds_audio_buffer* res = (struct ds_audio_buffer*)malloc(sizeof(struct ds_audio_buffer));

    sox_format_t* in = sox_open_read("/audio/1028-20100710-hne/wav/ar-01.wav", NULL, NULL, NULL);
    sox_effect_t * e;

    sox_signalinfo_t interm_signal;

    sox_signalinfo_t target_signal = {
      16000, // Rate
      1, // Channels
      16, // Precision
      SOX_UNSPEC, // Length
      NULL // Effects headroom multiplier
    };


    sox_encodinginfo_t target_encoding = {
    SOX_ENCODING_SIGN2, // Sample format
    16, // Bits per sample
    0.0, // Compression factor
    sox_option_default, // Should bytes be reversed
    sox_option_default, // Should nibbles be reversed
    sox_option_default, // Should bits be reversed (pairs of bits?)
    sox_false // Reverse endianness
  };


    sox_format_t* out = sox_open_memstream_write(&res->buffer, &res->buffer_size, &target_signal, &target_encoding, "raw", NULL);
    res->sample_rate = (int)out->signal.rate;

    sox_effects_chain_t* chain = sox_create_effects_chain(&in->encoding, &out->encoding);
    
    interm_signal = in->signal;

    char* sox_args[10];
    //input effect
    e = sox_create_effect(sox_find_effect("input"));
    sox_args[0] = (char*)in;
    assert(sox_effect_options(e, 1, sox_args) == SOX_SUCCESS);
    assert(sox_add_effect(chain, e, &interm_signal, &in->signal) ==
         SOX_SUCCESS);
    free(e);

    //tempo
    //e = sox_create_effect(sox_find_effect("tempo"));

    e = sox_create_effect(sox_find_effect("rate"));
    assert(sox_effect_options(e, 0, NULL) == SOX_SUCCESS);
    assert(sox_add_effect(chain, e, &interm_signal, &out->signal) ==
         SOX_SUCCESS);
    free(e);

    e = sox_create_effect(sox_find_effect("channels"));
    assert(sox_effect_options(e, 0, NULL) == SOX_SUCCESS);
    assert(sox_add_effect(chain, e, &interm_signal, &out->signal) ==
         SOX_SUCCESS);
    free(e);

    e = sox_create_effect(sox_find_effect("tempo"));
    std::string tempo_str = "0.5";
    sox_args[0] = (char*)tempo_str.c_str();
    assert(sox_effect_options(e, 1, sox_args) == SOX_SUCCESS);
    assert(sox_add_effect(chain, e, &interm_signal, &in->signal) ==
         SOX_SUCCESS);
    free(e);
    
    
    e = sox_create_effect(sox_find_effect("output"));
    sox_args[0] = (char*)out;
    assert(sox_effect_options(e, 1, sox_args) == SOX_SUCCESS);
    assert(sox_add_effect(chain, e, &interm_signal, &out->signal) ==
         SOX_SUCCESS);
    free(e);

    // Finally run the effects chain
    
    sox_flow_effects(chain, NULL, NULL);
    sox_delete_effects_chain(chain);

    sox_close(out);
    sox_close(in);
    

    cout << res->buffer_size;
    return 0;
}
