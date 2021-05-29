# Capstone 2: Character Recognition in Natural Images

In this project, I tackle the problem of character recognition in natural scenes - that is, images obtained through photographs of objects in the real world such as street signs, business signs, and other sources of text, rather than images of computer fonts or similarly normalized text. These “natural” images display a far greater degree of variability in text styles than computer fonts, so they pose a more significant challenge in the realm of computer vision; classical OCR methods do not perform nearly as well on natural images as they do on computer-generated text.

Milestone reports and a slide deck presentation may be found in the "Reports" folder, which go into significantly more detail on the strategy and execution of this preject.

# Summary of Methods Used

The model is built on a series of Convolutional Neural Networks (CNNs), which are a standard computational method used to identify key features in an input image. Those features may then be used to classify the image; in this case, the objective is to correctly classify images according to their pictured character.

I deploy a series of three CNNs to progressively achieve better accuracy in my predictions. Each of the three CNNs take different versions of the images as input, with the first taking 32x32 sized images, the second taking 64x64, and the third taking 128x128. By starting with low-resolution images, the model is forced to identify more prominent features of the images first, which helps to make more generalized predictions before moving on to identify the more granular differences between character types.

# Model Evaluation

The model was tested on 20% of the data using the holdout method. The computed loss and accuracy scores for the model are 0.4082 and 87.40%, respectively.

![evaluation](https://user-images.githubusercontent.com/59583730/120085555-c2c52d80-c09e-11eb-9924-bf31ade78696.png)

Below is a confusion matrix of the predictions as a heatmap. The y-axis labels mark the true (actual) classes, and the x-axis labels mark the predicted class assigned by the model. The middle diagonal signifies correct predictions, and other squares that are lit up show where the model made incorrect predictions.

![simple confusion matrix](https://user-images.githubusercontent.com/59583730/120085508-6d891c00-c09e-11eb-80af-dd7e69d27a52.png)

[Click here](https://user-images.githubusercontent.com/59583730/120085501-5a764c00-c09e-11eb-85e6-1f97c36c822e.png) for a detailed confusion matrix that shows the frequency of each class prediction, both as an absolute count and as a ratio of the number of predictions to the number of images in the (true) labeled class.

Overall, the model was quite successful, though there are patterns in the failure cases that may be used to inform future improvements.

# Failure Analysis

Below are the 10 classes that the model had the hardest time correctly identifying:

![lowest 10](https://user-images.githubusercontent.com/59583730/120085568-ec7e5480-c09e-11eb-881a-a9f39c0f7b9a.png)

One pattern that arises is confusion between uppercase and lowercase versions of letters, which understandably occurs more often when the two forms of a letter are visually similar (o and O, s and S, etc). There are also a few characters here that share visual similarities to other letters or numbers. The only character in this list that fits neither of these patterns is J, which has a number of nonstandard forms based on cursive writing that can appear very different from the standard printed J.

Data augmentation for under-represented classes could go a long way toward improving predictions. Additionally, since a full computer vision application would be taking full photographs as input rather than segmented images that isolate the characters, context provided by the full image could be leveraged to normalize capitalization or to assign greater weight to predictions that result in a word that appears in the app's dictionary, should it have one.

# Acknowledgements
[T. E. de Campos](http://personal.ee.surrey.ac.uk/Personal/T.Decampos/), B. R. Babu and [M. Varma](http://manikvarma.org/). [Character recognition in natural images](http://personal.ee.surrey.ac.uk/Personal/T.Decampos/papers/decampos_etal_visapp2009.pdf). In Proceedings of the International Conference on Computer Vision Theory and Applications (VISAPP), Lisbon, Portugal, February 2009.
