
Weights need to be sotred here in a form like `model_final_FIRST_INCISION_20190917.pth` 
- 20190917 - represents the date of the op, 
- FIRST_INCISION - represents the phase which is taken `detectron/phases.py`

For saving the validated results:
- create `images_predicted_multilabeled`
  - `full_prediction` here will be saved full predicted images with the standard detectron annotation
  - `layerwise_masks` here will be stored the predicted masks only. **This is important for the resection line construction**
    - to further predict the resection line create a dir named "1". Without this masks the construction of the resection line will not work. In future this can be done in-memory, without writing the mask to this location, but now it is **not** supported.
    - the "1" represents that the encoding from `nrrd_mapping.py` file, where `1` corresponds in `mapping_layers` to the `plane first incision` label, based on which will be constructed the resection line.
 - `layerwise` here will be placed the image in the following structure 
