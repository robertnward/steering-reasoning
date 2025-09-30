import os

if "HEAD_IDX" in os.environ and "PROJ_TYPE" in os.environ:
    from steering_reasoning.models.plugin import register_patch_head

    register_patch_head()
elif "ADD_PLACE" in os.environ:
    from steering_reasoning.models.plugin import register_add_place

    register_add_place()
