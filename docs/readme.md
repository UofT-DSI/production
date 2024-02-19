# A Note About the Course Materials

## Converting Slides from Markdown to PowerPoint.

Slides are written using regular markdown. You can convert them to PPTX file format using [pandoc](https://pandoc.org/):

```
pandoc -o output.pptx .\input.md --reference-doc=template.pptx
```

For example, 

```
pandoc -o production_1.pptx .\production_1_introduction.md --reference-doc='../slide_template.pptx'
```