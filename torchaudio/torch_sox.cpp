#include <torch/torch.h>

#include <sox.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <assert.h>

namespace torch {
namespace audio {
namespace {

struct ds_audio_buffer {
    char*  buffer;
    size_t buffer_size;
    int    sample_rate;
};


/// Helper struct to safely close the sox_format_t descriptor.
struct SoxDescriptor {
  explicit SoxDescriptor(sox_format_t* fd) noexcept : fd_(fd) {}
  SoxDescriptor(const SoxDescriptor& other) = delete;
  SoxDescriptor(SoxDescriptor&& other) = delete;
  SoxDescriptor& operator=(const SoxDescriptor& other) = delete;
  SoxDescriptor& operator=(SoxDescriptor&& other) = delete;
  ~SoxDescriptor() {
    sox_close(fd_);
  }
  sox_format_t* operator->() noexcept {
    return fd_;
  }
  sox_format_t* get() noexcept {
    return fd_;
  }

 private:
  sox_format_t* fd_;
};

void read_audio(
    SoxDescriptor& fd,
    at::Tensor output,
    int64_t number_of_channels,
    int64_t buffer_length) {
  std::vector<sox_sample_t> buffer(buffer_length);
  const int64_t samples_read = sox_read(fd.get(), buffer.data(), buffer_length);
  if (samples_read == 0) {
    throw std::runtime_error(
        "Error reading audio file: empty file or read failed in sox_read");
  }

  output.resize_({samples_read / number_of_channels, number_of_channels});
  output = output.contiguous();

  AT_DISPATCH_ALL_TYPES(output.type(), "read_audio_buffer", [&] {
    auto* data = output.data<scalar_t>();
    std::copy(buffer.begin(), buffer.begin() + samples_read, data);
  });
}

int64_t write_audio(SoxDescriptor& fd, at::Tensor tensor) {
  std::vector<sox_sample_t> buffer(tensor.numel());

  AT_DISPATCH_ALL_TYPES(tensor.type(), "write_audio_buffer", [&] {
    auto* data = tensor.data<scalar_t>();
    std::copy(data, data + tensor.numel(), buffer.begin());
  });

  const auto samples_written =
      sox_write(fd.get(), buffer.data(), buffer.size());

  return samples_written;
}
} // namespace

int read_audio_file_tempo_augment(const std::string& file_name, at::Tensor output, const std::string& new_tempo){

  struct ds_audio_buffer* res = (struct ds_audio_buffer*)malloc(sizeof(struct ds_audio_buffer));
  sox_format_t* in = sox_open_read(
          file_name.c_str(),
          /*signal=*/NULL,
          /*encoding=*/NULL,
          /*filetype=*/NULL);
  sox_effect_t * e;
  if (in==nullptr)
  {
      throw std::runtime_error("Error opening audio file");
  }

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

  //std::cout << "Input Sample rate : " << (int)in->signal.rate << std::endl;
  //std::cout << "Sample rate : " << res->sample_rate << std::endl;
  sox_effects_chain_t* chain = sox_create_effects_chain(&in->encoding, &out->encoding);

  interm_signal = in->signal;


  char* sox_args[10];
    //input effect
    e = sox_create_effect(sox_find_effect("input"));
    sox_args[0] = (char*)in;
    if(sox_effect_options(e, 1, sox_args) != SOX_SUCCESS)
        std::cout << "Coult not create effect options" << std::endl;
    if(sox_add_effect(chain, e, &interm_signal, &in->signal) !=
         SOX_SUCCESS)
        std::cout << "Could not add effect" << std::endl;
    free(e);

    e = sox_create_effect(sox_find_effect("rate"));
    if(sox_effect_options(e, 0, NULL) != SOX_SUCCESS)
        std::cout << "Coult not create effect rate" << std::endl;
    if(sox_add_effect(chain, e, &interm_signal, &out->signal) !=
         SOX_SUCCESS)
        std::cout << "Could not add effect rate" << std::endl;
    free(e);

    e = sox_create_effect(sox_find_effect("channels"));
    if(sox_effect_options(e, 0, NULL) != SOX_SUCCESS)
        std::cout << "Coult not create effect channels" << std::endl;
    if(sox_add_effect(chain, e, &interm_signal, &out->signal) !=
         SOX_SUCCESS)
        std::cout << "Could not add effect channels" << std::endl;
    free(e);


    e = sox_create_effect(sox_find_effect("tempo"));
    sox_args[0] = (char*)new_tempo.c_str();
    if (sox_effect_options(e, 1, sox_args) != SOX_SUCCESS)
        std::cout << "Coult not create effect options tempo" << std::endl;
    if (sox_add_effect(chain, e, &out->signal, &out->signal) !=
         SOX_SUCCESS)
        std::cout << "Coult not add effect tempo" << std::endl;
    free(e);
    
    
    e = sox_create_effect(sox_find_effect("output"));
    sox_args[0] = (char*)out;
    if(sox_effect_options(e, 1, sox_args) != SOX_SUCCESS)
        std::cout << "Coult not create effect options output" << std::endl;
    if(sox_add_effect(chain, e, &interm_signal, &out->signal) !=
         SOX_SUCCESS)
        std::cout << "Coult not add effect output" << std::endl;
    free(e);

  // Finally run the effects chain

  //std::cout << "Signal length " << signal_length << std::endl; 
  //std::cout << "res buffer  " << res->buffer_size << std::endl;

  sox_flow_effects(chain, NULL, NULL);

  std::vector<sox_sample_t> audio_buffer(interm_signal.length);
  const int64_t samples_read = sox_read(out, audio_buffer.data(), interm_signal.length);

  if (samples_read == 0) {
    throw std::runtime_error(
        "Error reading audio file: empty file or read failed in sox_read");
  }
 
  output.resize_({interm_signal.length, 1});
  output = output.contiguous();

  AT_DISPATCH_ALL_TYPES(output.type(), "read_audio_buffer", [&] {
    auto* data = output.data<scalar_t>();
    std::copy(audio_buffer.begin(), audio_buffer.begin() + interm_signal.length, data);
  }); 


  sox_delete_effects_chain(chain);
  
  sox_close(out);
  sox_close(in);
  free(res->buffer);
  free(res);

  return interm_signal.rate;
}

int read_audio_file(const std::string& file_name, at::Tensor output) {
  SoxDescriptor fd(sox_open_read(
      file_name.c_str(),
      /*signal=*/nullptr,
      /*encoding=*/nullptr,
      /*filetype=*/nullptr));
  if (fd.get() == nullptr) {
    throw std::runtime_error("Error opening audio file");
  }

  const int64_t number_of_channels = fd->signal.channels;
  const int sample_rate = fd->signal.rate;
  const int64_t buffer_length = fd->signal.length;
  if (buffer_length == 0) {
    throw std::runtime_error("Error reading audio file: unknown length");
  }

  read_audio(fd, output, number_of_channels, buffer_length);

  return sample_rate;
}

void write_audio_file(
    const std::string& file_name,
    at::Tensor tensor,
    const std::string& extension,
    int sample_rate) {
  if (!tensor.is_contiguous()) {
    throw std::runtime_error(
        "Error writing audio file: input tensor must be contiguous");
  }

  sox_signalinfo_t signal;
  signal.rate = sample_rate;
  signal.channels = tensor.size(1);
  signal.length = tensor.numel();
  signal.precision = 32; // precision in bits

#if SOX_LIB_VERSION_CODE >= 918272 // >= 14.3.0
  signal.mult = nullptr;
#endif

  SoxDescriptor fd(sox_open_write(
      file_name.c_str(),
      &signal,
      /*encoding=*/nullptr,
      extension.c_str(),
      /*filetype=*/nullptr,
      /*oob=*/nullptr));

  if (fd.get() == nullptr) {
    throw std::runtime_error(
        "Error writing audio file: could not open file for writing");
  }

  const auto samples_written = write_audio(fd, tensor);

  if (samples_written != tensor.numel()) {
    throw std::runtime_error(
        "Error writing audio file: could not write entire buffer");
  }
}
} // namespace audio
} // namespace torch

PYBIND11_MODULE(_torch_sox, m) {
  m.def(
      "read_audio_file",
      &torch::audio::read_audio_file,
      "Reads an audio file into a tensor");
  m.def(
  "read_audio_file_tempo_augment",
  &torch::audio::read_audio_file_tempo_augment,
  "Reads an audio file into a tensor with tempo augmentation");
  m.def(
      "write_audio_file",
      &torch::audio::write_audio_file,
      "Writes data from a tensor into an audio file");
}
