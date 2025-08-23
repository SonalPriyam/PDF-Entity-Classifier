import json
from pdf_processing import pdf_to_images, extract_text_blocks
from model_inference import load_model, predict_entities
from post_processing import rule_based_refinement
from visualization_cv import draw_boxes_cv


def main(pdf_path, output_path_pdf):
    print("Converting PDF to images...")
    images = pdf_to_images(pdf_path)

    print("Extracting text blocks...")
    text_blocks = extract_text_blocks(pdf_path)

    num_classes = 8  # customize this according to your classes
    print("Loading model...")
    processor, model, device = load_model(num_classes)

    print("Running model inference...")
    raw_results = predict_entities(processor, model, device, images, text_blocks)

    print("Applying rule-based refinement...")
    final_results = []
    annotated_images = []
    for page_img, page_entities in zip(images, raw_results):
        refined = rule_based_refinement(page_entities)
        final_results.append(refined)

        # Draw boxes and labels on image
        annotated_img = draw_boxes_cv(page_img, refined)
        annotated_images.append(annotated_img)

    print(f"Saving annotated PDF to {output_path_pdf} ...")
    annotated_images[0].save(
        output_path_pdf,
        save_all=True,
        append_images=annotated_images[1:]
    )

    print("Done.")

if __name__ == "__main__":
    import sys
    pdf_path = "data/Workbook.pdf" if len(sys.argv) < 2 else sys.argv[1]
    output_path_pdf = "outputs/annotated_output.pdf" if len(sys.argv) < 3 else sys.argv[2]
    main(pdf_path, output_path_pdf)
