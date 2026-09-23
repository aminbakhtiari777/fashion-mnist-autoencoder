# Fashion Vision Lab — Model Card

## Intended use

Educational comparison of supervised classification, image reconstruction, and generative modeling on Fashion-MNIST.

## Out-of-scope use

- Real-world fashion search or recommendation
- Product authentication
- Decisions affecting people
- Claims about commercial garment quality

## Dataset

Fashion-MNIST contains 70,000 grayscale 28×28 images across ten garment classes. It is useful for compact experiments but does not represent real photography, fabric behavior, luxury construction, cultural context, or contemporary collections.

## Evaluation

- Classification: accuracy, macro F1, and per-class recall
- Reconstruction: MSE, MAE, and PSNR
- Generation: sample diversity and pixel statistics

The generative metrics are lightweight collapse indicators. They are not substitutes for FID/KID, nearest-neighbor checks against training data, or structured human evaluation.

## Known limitations

- Historical notebooks were trained in separate sessions.
- Random seeds and dependency versions were not recorded in every early experiment.
- Fashion-MNIST labels are broad and visually ambiguous.
- Generated samples should not be presented as original commercial designs.

## Ethical presentation

Results must be labeled as benchmark experiments. Generated imagery must not be attributed to a real designer or brand without permission.
