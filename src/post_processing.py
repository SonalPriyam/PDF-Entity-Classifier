def rule_based_refinement(page_entities):
    """
    Refine initial predictions with deterministic rules.
    Input: List of dicts with `bbox`, `font_size`, `predicted_class`.
    Modify or override the predicted_class based on heuristics.
    """

    for entity in page_entities:
        x0, y0, x1, y1 = entity['bbox']
        font_size = entity.get('font_size', 0)

        # Example rule: if text near top and large font -> header
        if y0 < 100 and font_size > 10:
            entity['predicted_class'] = 'Header'
        
        # Example rule: if text near bottom -> footer
        # (assuming page height ~ 1120 px for example)
        if y1 > 1020:
            entity['predicted_class'] = 'Footer'

        # Additional rules for tables, figures, equations can be added here

    return page_entities
