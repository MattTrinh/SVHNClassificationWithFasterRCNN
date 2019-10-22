# Configuration class holding settings for the model
class Config:
    def __init__(self):
        # Image Settings
        self.min_side_size = 600
    
        # Anchor Settings
        # Use [8, 16, 32] if using 32x32 pre-processed images
        self.anchor_box_scales = [128, 256, 512]
        self.anchor_box_ratios = [0.5, 1.0, 2.0]
        
        # Classification categories
        self.categories = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        
        # RPN Settings
        self.max_proposals = 300
        self.padding = 1
        # Use 4 if using 32x32 pre-processed images
        self.stride = 16
        self.overlap_thresh = 0.7