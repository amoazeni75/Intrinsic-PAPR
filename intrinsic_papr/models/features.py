"""Slicing the per-pixel feature map produced by proximity attention."""



def extract_features_from_feature_map(features_map, features_dim):
    """
    Take the last `features_dim` channels of the feature map.

    features_map: [B, H, W, C]
    features_dim: int: the size of the slice of the feature map that we need to extract
    """
    features_map_dim = features_map.shape[-1]
    assert features_map_dim >= features_dim
    return features_map[..., features_map_dim - features_dim :]
