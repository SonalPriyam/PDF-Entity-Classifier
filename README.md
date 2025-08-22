In this project, we built a pipeline to accurately classify different entities in a PDF document such as headers, tables, text, figures, equations, charts, and footers—by combining deep learning with some logic. The  summary of what we did is :



First step :-PDF Preprocessing & Rendering
In this,we converted each PDF page to images using pdf2image with Poppler installed. This enabled us to apply visual document models that work on images.

We also extracted text blocks with bounding boxes and font sizes from the PDF using PyMuPDF, so that both the content and layout information are available.

Second Step:- Deep Learning Layout Detection

We loaded a pretrained LayoutLMv3 model to classify different layout entities based on the combined visual and textual features.

Step 3 :- Rule-Based Post-Processing
In this, we implemented deterministic rules based on location, font size, and layout geometry to correct the model predictions.

Step 4 :- We got our OutPut