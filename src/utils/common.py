def create_filter_func(filter_expr: str):
    """Convert string filter to function"""
    if not filter_expr:
        return None

    def filter_func(entity):
        # Example: 'frame_id like "video1#%"'
        if "like" in filter_expr:
            parts = filter_expr.split("like")
            field = parts[0].strip()
            pattern = parts[1].strip().strip("\"'")

            if field in entity:
                value = str(entity[field])
                if pattern.endswith("%"):
                    return value.startswith(pattern[:-1])
                elif pattern.startswith("%"):
                    return value.endswith(pattern[1:])
                else:
                    return pattern in value

        return True

    return filter_func
