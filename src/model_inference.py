import pytesseract

# Set Tesseract path 
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
import torch.nn.functional as F

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
    Perform entity classification on each page image.
    Returns list of dicts with text, bbox, predicted_class, and confidence.
    """
    results = []
    
    # Map numeric label indices to class names
    class_names = [
        "Header", "Table", "Text", "Figure", "Equation", "Chart", "Footer", "Other"
    ]
    
    for img, blocks in zip(images, text_blocks):
        encoding = processor(img, return_tensors="pt")
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)
        
        logits = outputs.logits.squeeze(0).detach().cpu()  
        probs = F.softmax(logits, dim=1)  
        predictions = torch.argmax(logits, dim=1).tolist()
        confidences = probs.max(dim=1).values.tolist()

        page_result = []
        for block, pred, conf in zip(blocks, predictions, confidences):
            page_result.append({
                "text": block['text'],
                "bbox": block['bbox'],
                "predicted_class": class_names[pred] if pred < len(class_names) else "Unknown",
                "confidence": conf
            })
        results.append(page_result)
    return results

