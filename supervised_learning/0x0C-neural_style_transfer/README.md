                                                             Neural Style Transfer



Neural style transfer is an optimization technique used to take three images, a content image, a style reference image (such as an artwork by a famous painter), and the input image you want to style — and blend them together such that the input image is transformed to look like the content image, but “painted” in the style of the style image.

![image](https://user-images.githubusercontent.com/49324230/135998517-8f8bf8a5-30a8-4204-acc3-700efe110e9c.png)
![image](https://user-images.githubusercontent.com/49324230/135999667-e8ee3ccc-9517-4d8f-a6ae-ae17733f6c22.png)

 Two images are input to the neural network i.e. a content image and a style image. Our motive here is to generate a mixed image that has contours of the content image and texture, color pattern of the style image. We do this by optimizing several loss functions.
