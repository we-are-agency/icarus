use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use ringbuf::{Consumer, Producer, RingBuffer};

pub const FFT_SIZE: usize = 2048;
const RING_SIZE: usize = FFT_SIZE * 8;

pub struct AudioCapture {
    _stream: cpal::Stream, // must stay alive for the duration of capture
    pub consumer: Consumer<f32>,
    pub sample_rate: u32,
}

fn preferred_input_device(host: &cpal::Host) -> Option<cpal::Device> {
    use cpal::device_description::{DeviceType, InterfaceType};
    use cpal::traits::DeviceTrait;

    let devices: Vec<_> = host.input_devices().ok()?.collect();

    println!("Available input devices:");
    for d in &devices {
        if let Ok(desc) = d.description() {
            println!("  • {desc}");
        }
    }

    const LOOPBACK_NAMES: &[&str] = &[
        "Background Music",
        "BlackHole",
        "Loopback",
        "Soundflower",
        "CABLE Output",
    ];

    // Prefer virtual/loopback devices (by type or by well-known name)
    let loopback = devices.into_iter().find(|d| {
        d.description()
            .map(|desc| {
                desc.device_type() == DeviceType::Virtual
                    || desc.interface_type() == InterfaceType::Virtual
                    || LOOPBACK_NAMES.iter().any(|k| desc.name().contains(k))
            })
            .unwrap_or(false)
    });

    if let Some(ref d) = loopback {
        if let Ok(desc) = d.description() {
            println!("Using system audio device: {desc}");
        }
    } else {
        println!("No virtual/loopback device found — falling back to default input (mic).");
        println!(
            "Install BlackHole for system audio: https://github.com/ExistentialAudio/BlackHole"
        );
    }

    loopback.or_else(|| host.default_input_device())
}

impl AudioCapture {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let host = cpal::default_host();
        let device = preferred_input_device(&host).ok_or("no input device found")?;

        let config = device.default_input_config()?;
        let sample_rate = config.sample_rate();

        let (producer, consumer) = RingBuffer::<f32>::new(RING_SIZE).split();

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => build_stream::<f32>(&device, &config.into(), producer),
            cpal::SampleFormat::I16 => build_stream::<i16>(&device, &config.into(), producer),
            cpal::SampleFormat::U16 => build_stream::<u16>(&device, &config.into(), producer),
            cpal::SampleFormat::I32 => build_stream::<i32>(&device, &config.into(), producer),
            cpal::SampleFormat::F64 => build_stream::<f64>(&device, &config.into(), producer),
            fmt => return Err(format!("unsupported sample format: {fmt:?}").into()),
        }?;

        stream.play()?;

        Ok(Self {
            _stream: stream,
            consumer,
            sample_rate,
        })
    }
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut producer: Producer<f32>,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = config.channels as usize;
    device.build_input_stream(
        config,
        move |data: &[T], _| {
            // Downmix interleaved multi-channel input to mono before pushing.
            // On stereo loopback this prevents L,R,L,R,... from corrupting the FFT.
            let ch = channels.max(1);
            for frame in data.chunks(ch) {
                let mono = frame.iter().map(|&s| s.to_sample::<f32>()).sum::<f32>() / ch as f32;
                let _ = producer.push(mono);
            }
        },
        |err| eprintln!("audio stream error: {err}"),
        None,
    )
}
