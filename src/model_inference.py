from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch

def load_model(num_labels):
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=num_labels
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device

def predict_entities(processor, model, device, images, text_blocks):
    """
    Perform entity classification on each page.
    images: list of PIL images
    text_blocks: corresponding list of text blocks with bbox info
    """
    results = []
    for img, blocks in zip(images, text_blocks):
        # Prepare model inputs for LayoutLMv3 here:
        # Tokenization & layout mapping must be done per model requirements
        # For simplicity this is a placeholder example:
        encoding = processor(img, return_tensors="pt")
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)
        
        logits = outputs.logits.detach().cpu()
        predictions = torch.argmax(logits, dim=2).squeeze().tolist()
        
        # Map predictions back to text_blocks - here simplified:
        page_result = []
        for block, pred in zip(blocks, predictions):
            page_result.append({
                "text": block['text'],
                "bbox": block['bbox'],
                "predicted_class": pred
            })
        results.append(page_result)
    return results
