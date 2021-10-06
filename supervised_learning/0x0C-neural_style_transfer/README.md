                                                             Neural Style Transfer



Neural style transfer is an optimization technique used to take three images, a content image, a style reference image (such as an artwork by a famous painter), and the input image you want to style — and blend them together such that the input image is transformed to look like the content image, but “painted” in the style of the style image.

![image](https://user-images.githubusercontent.com/49324230/136174853-58e970cf-9e09-4847-815c-c081c0d1c54e.png)


Two images are input to the neural network i.e. a content image and a style image. Our motive here is to generate a mixed image that has contours of the content image and texture, color pattern of the style image. We do this by optimizing several loss functions.

![image](https://user-images.githubusercontent.com/49324230/135998517-8f8bf8a5-30a8-4204-acc3-700efe110e9c.png)


The loss function for the content image minimizes the difference of the features activated for the content image corresponding to the mixed image (which initially is just a noise image that gradually improves) at one or more layers. This preserves the contour of the content image to the resultant mixed image.
Whereas the loss function for the style image minimizes the difference between so-called Gram-matrices between style image and the mixed image. This is done at one or more layers. The usage of the Gram matrix is it identifies which features are activated simultaneously at a given layer. Then we mimic the same behavior to apply it to the mixed image.
Using TensorFlow, we update the gradient of these combined loss functions of content and style image to a satisfactory level. Certain calculations of Gram matrices, storing intermediate values for efficiency, loss function for denoising of images, normalizing combined loss function so both image scale relative to each other.


![image](https://user-images.githubusercontent.com/49324230/135999667-e8ee3ccc-9517-4d8f-a6ae-ae17733f6c22.png)

