## Project overview
### Directories
- [/data](https://github.com/fivebillionmph/be223c/tree/master/data)
  - This is where all the variable or generated data is kept.  This includes the classification models, miniVGG model, automatic lung segmentation model, model test results and the lung images (original images, lesion segmentations, patches, etc).
  - This directory is not as organized, it has unused models and images because data was swapped out and interchanged.
  - It is gitignored, but the data is packaged in the zip file submitted to CCLE.
- [/scripts](https://github.com/fivebillionmph/be223c/tree/master/scripts)
  - Simple scripts for starting the web server and getting the installed Python modules
- [/src](https://github.com/fivebillionmph/be223c/tree/master/src)
  - Python scripts for running the server, segmenting, testing models, training the miniVGG, etc.
- [/src/mod](https://github.com/fivebillionmph/be223c/tree/master/src/mod)
  - Shared modules used by scripts in /src.
- [/src2](https://github.com/fivebillionmph/be223c/tree/master/src2)
  - Other scripts for generating models or segmenting.
- [/web](https://github.com/fivebillionmph/be223c/tree/master/web)
  - HTML templates and static CSS, JS and images
