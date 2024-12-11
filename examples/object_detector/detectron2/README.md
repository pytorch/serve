# Object Detection using TorchServe and Detectron2

## Overview

This folder leverages **TorchServe** to deploy a Detectron2-based object detection model using a custom handler. It provides scalable and efficient object detection capabilities with support for both CPU and GPU environments.

---

## Table of Contents

1. [Pre-requirements](#pre-requirements)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Documentation](#documentation)
5. [Contributors](#contributors)

---

## Pre-requirements

- **Python 3.8 or higher** (tested on Python 3.10.15).

---

## Installation

Follow these steps to set up the project:

1. Clone the repository:

   ```bash
   git clone https://github.com/pytorch/serve.git
   ```

2. Make sure the terminal's current directory is set to the folder where this README file is located:

   ```bash
   cd serve/examples/object_detector/detectron2
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   pip install git+https://github.com/facebookresearch/detectron2.git && pip install numpy==1.21.6
   ```

---

## Usage

Refer to the [Documentation](#documentation) for detailed usage instructions.

---

## Documentation

For detailed information on using TorchServe and Detectron2 for object detection, refer to the documentation provided in the [Upstart Commerce Blog](https://upstartcommerce.com/blogs/).

---

## Contributors

- **[Muhammad Mudassar](https://github.com/Mudassar-MLE)**  
  - [LinkedIn](https://www.linkedin.com/in/muhammad-mudassar-a65645192/)  
  - [Email](mailto:mmudassards@gmail.com)
---
