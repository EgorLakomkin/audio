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

int read_audio_file_augment(const std::string& file_name, at::Tensor output, const std::vector<std::string>& augment_params){


  char*  buffer;
  size_t buffer_size;
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

  sox_format_t* out = sox_open_memstream_write(&buffer, &buffer_size, &in->signal, &in->encoding, "raw", NULL);

  //std::cout << "Input Sample rate : " << (int)in->signal.rate << std::endl;
  //std::cout << "Sample rate : " << res->sample_rate << std::endl;
  sox_effects_chain_t* chain = sox_create_effects_chain(&in->encoding, &out->encoding);


  char* sox_args[10];
    //input effect
    e = sox_create_effect(sox_find_effect("input"));
    sox_args[0] = (char*)in;
    if(sox_effect_options(e, 1, sox_args) != SOX_SUCCESS)
        std::cout << "Coult not create effect options" << std::endl;
    if(sox_add_effect(chain, e,  &in->signal, &in->signal) !=
         SOX_SUCCESS)
        std::cout << "Could not add effect" << std::endl;
    free(e);

    std::vector<std::string>::const_iterator it;

    for(it = augment_params.begin(); it != augment_params.end(); it++)
    {
        const std::string& aug_param = *it;
        if ((aug_param.compare("volume") == 0) || (aug_param.compare("tempo") == 0) || (aug_param.compare("pitch") == 0) || (aug_param.compare("speed")==0) || (aug_param.compare("gain") == 0))
        { 
            //std::cout << "add effect" << aug_param << std::endl;
            e = sox_create_effect(sox_find_effect((char*)aug_param.c_str()));
            //e->global_info = sox_get_effects_globals();
            e->global_info->global_info->verbosity = 0;            

            it++;
            const std::string& effect_value = *it;
            sox_args[0] = (char*)(effect_value.c_str());
            if (sox_effect_options(e, 1, sox_args) != SOX_SUCCESS)
                std::cout << "Coult not create effect options" << std::endl;
            if (sox_add_effect(chain, e,  &in->signal,  &in->signal) !=
                 SOX_SUCCESS)
                std::cout << "Coult not add effect" << std::endl;
            free(e);
        }
    }
    
    e = sox_create_effect(sox_find_effect("output"));
    sox_args[0] = (char*)out;
    if(sox_effect_options(e, 1, sox_args) != SOX_SUCCESS)
        std::cout << "Coult not create effect options output" << std::endl;
    if(sox_add_effect(chain, e,  &in->signal,  &in->signal) !=
         SOX_SUCCESS)
        std::cout << "Coult not add effect output" << std::endl;
    free(e);

    sox_flow_effects(chain, NULL, NULL);


    int sample_rate = out->signal.rate;
    static const size_t maxSamples=2048;
    sox_sample_t samples[maxSamples];

    std::vector<sox_sample_t> audio_buffer;
    for (size_t r; 0 != (r=sox_read(out,samples,maxSamples));)
        for(int i=0;i<r ;i++)
            audio_buffer.push_back(samples[i]);

    output.resize_({audio_buffer.size(), 1});
    output = output.contiguous();


    AT_DISPATCH_ALL_TYPES(output.type(), "read_audio_buffer", [&] {
        auto* data = output.data<scalar_t>();
        std::copy(audio_buffer.begin(), audio_buffer.begin() + audio_buffer.size(), data);
    });
    //TODO: CLEAR MEMORY
    //free(buffer);
    sox_delete_effects_chain(chain);
    sox_close(out);
    sox_close(in);
    free(buffer);
    return sample_rate;
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
  "read_audio_file_augment",
  &torch::audio::read_audio_file_augment,
  "Reads an audio file and applies augmentations");
  m.def(
      "write_audio_file",
      &torch::audio::write_audio_file,
      "Writes data from a tensor into an audio file");
}
