# goat library

This library contains code and tools useful in computer vision research. The goal is to keep high-level functions readily available for use across projects. We've organized the repo into the following components.

### cache
- In computer vision, often times we are dealing with large amounts of data. Loading objects in Jupyter Notebooks to cache in RAM is a reasonable solution to slow loading times, but this is restrictive. This component is meant to provide a light wrapper server to keep items in RAM and quickly fetch data with network requests.

### edit
- This contains commonly used code for drawing polygons on images, drawing segmentation results, high-level cropping, and more.

### mturk
- Many tasks in computer vision require annotating data, whether for creating a dataset or for quick human analysis and surveys of results. Rewriting code for the standard abstractions of using the boto3 MTurk library is tedious, so we keep high-level abstractions here.

### plot
- Matplotlib and Plotly are great for plots, but we often repeat setting up desired configurations commonly used in research papers. Here we keep high-level abstractions with for dealing with data and making plots.

### view
- A huge component of computer vision is looking at images to understand results. However, as most computer vision research is conducted in Python, this process becomes tedious for many reasons. It's difficult to load images and visualize effectively, without designing custom configs. This becomes repetitive across projects, so here we keep our favorite ways to visualize images, utilizing the browser and web development (HTML/CSS/JavaScript).