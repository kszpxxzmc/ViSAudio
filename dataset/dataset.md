# BiAudio

## Data Availability Statement
We are committed to maintaining transparency and compliance in our data collection and sharing methods. Please note the following:

- **Publicly Available Data**: The data utilized in our studies is publicly available. We do not use any exclusive or private data sources.

- **Data Sharing Policy**: Our data sharing policy aligns with precedents set by prior works, such as [InternVid](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid), [Panda-70M](https://snap-research.github.io/Panda-70M/) 
, and [Miradata](https://github.com/mira-space/MiraData). Rather than providing the original raw data, we only supply the YouTube video IDs necessary for downloading the respective content.

- **Usage Rights**: The data released is intended exclusively for research purposes. Any potential commercial usage is not sanctioned under this agreement.

- **Compliance with YouTube Policies**: Our data collection and release practices strictly adhere to YouTube’s data privacy policies and fair of use policies. We ensure that no user data or privacy rights are violated during the process.

- **Data License**: The dataset is made available under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

### Clarifications
- The BiAudio dataset is only available for informational purposes only. The copyright remains with the original owners of the video and audio.
- All videos and audios of the BiAudio dataset are obtained from the Internet which is not the property of our institutions. Our institution is not responsible for the content or the meaning of these videos and audios.
- You agree not to reproduce, duplicate, copy, sell, trade, resell, or exploit for any commercial purposes, any portion of the videos and audios, and any portion of derived data. You agree not to further copy, publish, or distribute any portion of the BiAudio dataset.

## Dataset Overview
**Note:** The [BiAudio](https://huggingface.co/datasets/Dubhe-zmc/BiAudio) dataset comprises 97K perspective video clips with corresponding binaural spatial audio and fine-grained textual captions. Each data sample includes metadata such as YouTubeID, StartTime. In addition to the raw data, the processed dataset features captions and camera rotation trajectories.

### Dataset Metadata
The files [dataset/train_valid.tsv](./train_valid.tsv) and [dataset/test_valid.tsv](./test_valid.tsv) contain the following columns:
- **id**: Each ID follows the format `{YouTubeID}_{StartTime}`.
  - **YouTubeID**: The YouTube ID of the original video (e.g., `yhFN_xVmNsI`). The source video can be accessed at `https://www.youtube.com/watch?v={YouTubeID}`.
  - **StartTime**: The starting time of the video segment in seconds. The corresponding end time is defined as `EndTime = StartTime + 8`. Video, audio, and camera rotation trajectories cover the same **8-second** segment.
- **label**: A textual caption describing the video-audio segment, including both **visible** and **invisible** sound events.

**Example Data Entry**
| id | label |
|---------|-----------|
| yhFN_xVmNsI_200 | Visible sound: Waves crashing, Invisible sound: Continuous ambient ocean noise |


### Dataset Format
During training, we organize the data in the following structure:
```markdown
BiAudio // BiAudio Dataset
├── caption
│   ├── <split>_valid.tsv
│   │       // Dataset metadata containing captions describing each video-audio pair.
│   │       // Includes:
│   │       //   - Visible (Visible Caption)
│   │       //   - Invisible (Invisible Caption)
├── video // The video folder is constructed by the `Dataset Construction` script
│   ├── <split>
│   │   ├── <id>.mp4
│   │   │       // 8-second perspective video clips with binaural audio
├── (optional) metadata
│   ├── <split>
│   │   ├── <id>_rpy.json
│   │   │       // 8-second camera rotation trajectories, fps aligned with the video.
```

## Dataset Construction

### Data Collection 
We download paired first-order ambisonics (FOA) audio and 360° video using the [Sphere360 script](https://github.com/liuhuadai/OmniAudio/blob/master/Sphere360/toolset/crawl/download/download_list.sh) from [OmniAudio](https://github.com/liuhuadai/OmniAudio). 

**Note:** Due to YouTube policy restrictions, we perform the downloading process in a loop to ensure that all UIDs are successfully retrieved. All files are saved in the `.webm` format.

### Data Processing
Here are the instructions to reproduce the BiAudio dataset using the data processing scripts.

**Note:** Samples not meeting the requirements have been excluded from the metadata. The `optional command` below is intended for use when constructing a custom dataset.
#### 1. Data Sampling
```bash
python prepare_dataset.py clip # Crop aligned 8-second video and audio clips
python prepare_dataset.py nfov # Extract field-of-view videos and corresponding binaural audios
```

#### 2. Data Annotation
```bash
(optional) python prepare_caption.py # Generate captions describing each video-audio pair. Format: Visible sound: <audible sound>[, Invisible sound: <background sound>]
```

#### 3. Data Curation 
```bash
python format_dataset.py # Resample binaural audio to 44.1 kHz and merge the audio and video files into 8-second audiovisual clips
(optional) python filter_dataset.py # Discard samples with minimal left-right channel differences
```

## License
The `BiAudio` dataset is available under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Please ensure proper attribution when using the dataset in research or other projects.

## Citation
If you use `BiAudio` in your research, please cite it as follows:

```bibtex
@misc{zhang2025visaudioendtoendvideodrivenbinaural,
    title={ViSAudio: End-to-End Video-Driven Binaural Spatial Audio Generation}, 
    author={Mengchen Zhang and Qi Chen and Tong Wu and Zihan Liu and Dahua Lin},
    year={2025},
    eprint={2512.03036},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2512.03036}, 
}
```
