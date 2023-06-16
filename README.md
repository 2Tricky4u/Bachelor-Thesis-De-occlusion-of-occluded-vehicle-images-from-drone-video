<!-- PROJECT LOGO -->
<br />
<div align="center" id="readme-top">
  <a href="https://www.epfl.ch/labs/luts/">
    <img src="Images/luts_logo.png" alt="Logo" width="237" height="80">
  </a>

<h3 align="center">Bachelor Thesis: De-occlusion of occluded vehicle images from drone video</h3>

  <p align="center">
    An adventure through Inpainting hidden part of vehicles with Deep Learning!
    <br />
    <a href="googledrive"><strong>Explore the report »</strong></a> <!-- TODO: Add link to report -->
    <br />
    <br />
    <a href="#midsem">View Mid-semester results</a>
    ·
    <a href="#endsem">View End-semester results</a>
    ·
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

![Inpainting Screen Shot](./Images/project_pres.png)

In urban traffic analysis, the accurate detection of vehicles plays a crucial role in generating reliable statistics for various city management applications. However, occlusions occurring in densely populated city environments pose significant challenges to vehicle detection algorithms, leading to reduced detection rates and compromised data accuracy. To try to address this issue, we compared two of the novel models that leverage machine learning techniques for inpainting occluded vehicles, with the goal of improving the overall detection rate and enhancing the reliability of city statistics.


De-occlusion involves a two-step process: 
* Occlusion detection (segmentation)
* Inpainting (image completion) 

First, an occlusion detection algorithm is employed to identify regions within the traffic scene that contain occluded vehicles. 
<br>Second, is an Inpainting model that given the occluded part, completes this hidden image fraction. 

<b>We exclusively focused on the inpainting part of images for this project.</b>

One machine learning model, namely Repaint, is trained using only a dataset of non-occluded vehicle images and the second, namely AOT-GAN, also needs masks of the occluded part. The models learn to inpaint the occluded regions based on the available visual information. By leveraging contextual cues and vehicle appearance patterns, the model should effectively generate plausible completions of the occluded regions, restoring the missing vehicle details.

We wanted to evaluate the proposed method on a comprehensive dataset of UAV point of view single vehicles in urban traffic scenes. Finetune it with another dataset of the LUTS lab. Then comparing the ground truth image with the inpainted image to compare the results. An interesting future evaluation could be comparing the detection performance before and after applying the inpainting technique.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

Here are the major frameworks/libraries used in our project.

* [![PyTorch][PyTorch.org]][PyTorch-url]
* [![Numpy][Numpy.org]][Numpy-url]
* [![OpenCV][OpenCV.org]][OpenCV-url]
* [![Cairo][Cairo.org]][Cairo-url]

This project was based on the following GitHub repositories.

Main Machine Learning Models used:
* [![Repaint][Git-repo]][Repaint-url] <i> <a href="https://github.com/andreas128/RePaint">RePaint: Inpainting using Denoising Diffusion Probabilistic Models</a></i>
* [![Guided-Diffusion][Git-repo]][Guided-Diffusion-url] <i> <a href="https://github.com/openai/guided-diffusion">Guided Diffusion (Repaint training pipeline)</a></i>
* [![AOT-GAN][Git-repo]][AOT-GAN-url] <i> <a href="https://github.com/researchmm/AOT-GAN-for-Inpainting">AOT-GAN for High-Resolution Image Inpainting</a></i>

Evaluation Metrics:
* [![Inpainting-Evaluation-Metrics][Git-repo]][Inpainting-Evaluation-Metrics-url] <i> <a href="https://github.com/SayedNadim/Image-Quality-Evaluation-Metrics">Image Quality Evaluation Metrics</a></i>

2D Shape Generator:
* [![2D-Shape-Generator][Git-repo]][2D-Shape-Generator-url] <i> <a href="https://github.com/TimoFlesch/2D-Shape-Generator">2D-Shape-Generator</a></i>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps. <br> We need to setup, for both model, the environment, download the datasets and the models, and then run the code.

### Prerequisites

Get the latest version of Python and pip installed on your machine. <br>
We also highly suggest the use of conda to manage the virtual environments. <br>
We recommend using a virtual environment to run the code of each model individually. <br>
To fetch the code from GitHub, you need to have git installed on your machine. <br>
You can download this repo from the following command:

```sh
  git clone https://github.com/2Tricky4u/Bachelor-Thesis-De-occlusion-of-occluded-vehicle-images-from-drone-video
  ```

### Installation

#### Repaint (Only inferring)

1. Go to the Repaint folder
   ```sh
   cd Models/Repaint
   ```
2. When in the appropriate virtual environment, install the requirements:
   ```sh
   pip install numpy torch blobfile tqdm pyYaml pillow  
   ```
   You should be ready to use Repaint if your machine has a GPU with CUDA support. <br> Go to the <a href="#repaint-usage">Repaint Usage</a> section to see how to use it.

#### Guided Diffusion (Training Repaint)

1. Go to the Guided Diffusion folder
   ```sh
   cd Models/Guided-Diffusion
   ```
2. When in the appropriate virtual environment, install the requirements:
   ```sh
   pip install -e .
   ```
3. You also need the following Message Passing Interface (MPI) library:
   ```sh
   pip install mpi4py
   ```   
   You should be ready to use Guided Diffusion and train a model for Repaint (if your machine has a GPU with CUDA support.) <br> Go to the <a href="#guided-usage">Guided Diffusion Usage</a> section to see how to use it and launch training.   

#### AOT-GAN

1. Go to the AOT-GAN folder
   ```sh
   cd Models/AOT-GAN
   ```
   
2. With conda, create a new virtual environment and install the requirements:
   ```sh
   conda env create -f environment.yml
   ```
   
3. Activate the environment
   ```sh
   conda activate inpainting
   ```
   You should be ready to use AOT-GAN for inference and training (if your machine has a GPU with CUDA support.) <br> Go to the <a href="#AOT-usage">AOT-GAN Usage</a> section to see how to use it.

#### Evaluation Metrics

1. Go to the Evaluation Metrics folder
   ```sh
   cd Metrics
   ```
2. When in the appropriate virtual environment, install the requirements:
   ```sh
   pip install piq
   ```
    You should be ready to use the Evaluation Metrics. <br> Go to the <a href="#metrics-usage">Evaluation Metrics Usage</a> section to see how to use it.
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Xavier Ogay - [website](https://git.xavierogay.ch/) - xavier.ogay@epfl.ch

Mahmoud Dokmak - mahmoud.dokmak@epfl.ch

Project Link: [https://github.com/2Tricky4u/Bachelor-Thesis-De-occlusion-of-occluded-vehicle-images-from-drone-video](https://github.com/2Tricky4u/Bachelor-Thesis-De-occlusion-of-occluded-vehicle-images-from-drone-video)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Report

<object data="report.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="report.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="report.pdf">Download PDF</a>.</p>
    </embed>
</object>

## Results
#### Mid-Semester Sample Results
![Mid-Semester-Results][Mid-Semester-Results]<a id="midsem"></a>

#### End-Semester Sample Results
![End-Semester-Results][End-Semester-Results]<a id="endsem"></a>
<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Pytorch.org]: https://img.shields.io/badge/PyTorch-FF0000?style=for-the-badge&logo=PyTorch&logoColor=white
[Pytorch-url]: https://pytorch.org/
[Numpy.org]: https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white
[Numpy-url]: https://numpy.org/
[Cairo.org]: https://img.shields.io/badge/Cairo-000000?style=for-the-badge&logo=cairo&logoColor=white
[Cairo-url]: https://www.cairographics.org/
[OpenCV.org]: https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white
[OpenCV-url]: https://opencv.org/
[Git-repo]: https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white
[Repaint-url]: https://github.com/andreas128/RePaint
[AOT-GAN-url]: https://github.com/researchmm/AOT-GAN-for-Inpainting
[Inpainting-Evaluation-Metrics-url]: https://github.com/SayedNadim/Image-Quality-Evaluation-Metrics
[2D-Shape-Generator-url]: https://github.com/TimoFlesch/2D-Shape-Generator
[Guided-Diffusion-url]: https://github.com/openai/guided-diffusion
[Mid-Semester-Results]: Images/Huge.png
