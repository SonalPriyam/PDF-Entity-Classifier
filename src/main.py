import json
from pdf_processing import pdf_to_images, extract_text_blocks
from model_inference import load_model, predict_entities
from post_processing import rule_based_refinement

def main(pdf_path, output_path):
    print("Converting PDF to images...")
    images = pdf_to_images(pdf_path)

    print("Extracting text blocks...")
    text_blocks = extract_text_blocks(pdf_path)

    num_classes = 10  # adjust according to your label set
    print("Loading model...")
    processor, model, device = load_model(num_classes)

    print("Running model inference...")
    raw_results = predict_entities(processor, model, device, images, text_blocks)

    print("Applying rule-based refinement...")
    final_results = []
    for page_entities in raw_results:
        refined = rule_based_refinement(page_entities)
        final_results.append(refined)

    print(f"Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)

    print("Done.")

if __name__ == "__main__":
    import sys
    pdf_path = "data/Workbook.pdf" if len(sys.argv) < 2 else sys.argv[1]
    output_path = "outputs/results.json" if len(sys.argv) < 3 else sys.argv
    main(pdf_path, output_path)

