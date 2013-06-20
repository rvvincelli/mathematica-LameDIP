(* ::Package:: *)

(*:Title:Lame DIP*)

(* :Author: Riccardo V. Vincelli, Universita' di Milano-Bicocca *)
(* :Mathematica Version: 7.0 *) 
(* :Package Version: 1.0 *) 
(* :Context: LameDIP`*)
(* :Summary:
   See README
*)          
(* :History:
I put together stuff I wrote
*)
(* :Updates:
-
*)
(* Botherings? Mail:
Riccardo V. Vincelli - r.vincelli@campus.unimib.it
*)
(*License: See README

*)

		(**Lame DIP**)


BeginPackage["LameDIP`"]

Print[Date[][[1]],"-",Date[][[2]],"-",Date[][[3]]]
Print["package by: Riccardo aka reim Vincelli"]
Print["Type ?LameDIP`* to visualize package functions"]
Print["Now loading ..."]

Unprotect[
BayerCFA,
Blueprint,
BCFAR,
BCFAG,
BCFAB,
BlackWhitePattern,
CMYKColorSeparate,
CMYKDominantAdd,
ColoredWhiteNoise,
ComicbookEffect,
CompoundEye,
ContourTrace,
EdgePlay,
EffettoPentito,
FlipperScreen,
FourierPhase,
FourierSpectrum,
GaussianScale,
GrayscaleClassicAnaglyph,
HSVToRGB,
Kaleidoscope,
ImageLineIntegralConvolution,
ImagePadSize,
ImagePlanes,
ImageShear,
ImageSignal,
Interlace,
IsOdd,
LineOverlay,
MeanSquareError,
Moroney,
NeonGlow,
PeakSignalToNoiseRatio,
Pixellate,
Pointillize,
RandomGradient,
RCColorSeparate,
RGColorSeparate,
RGBColorSeparate,
RGBDominantAdd,
RGBThreshold,
rgbthreshold,
RGBThreshold2,
rgbthreshold2,
RGBToHSV,
RGBToRC,
RGBToRG,
RGBToYCbCr,
RGBToYDbDr,
RGBToYIQ,
RGBToYUV,
RYBColorSeparate,
RMSContrast,
ShadeGradient,
ShadeIndexedColorTable,
ShadePhysicalGradient,
ShadeSingleColor,
SpraycanEffect,
Swirl,
ValueNoise,
YCbCrToRGB,
YDbDrToRGB,
YIQToRGB,
YUVToRGB,
WhiteBlackPattern,
WorleyNoise
]

(*BayerCFA*)
BayerCFA::usage = "This function transforms the input image in something like its raw version, fresh output from the array of sensors of a camera whose arrangement
follows the Bayer Color Filter Array standard. The algorithm is simple, decompose the image into the three components and multiply each by a proper mask to eliminate
(set to zero) the number of values needed, and then recombine."

(*Blueprint*)
Blueprint::usage = "The blueprint is a well known but substantially simple effect: an edge image is overlaid on a cyan-bluish monochrome and then a grid is drawn over."

(*CMYKColorSeparate*)
CMYKColorSeparate::usage = "CMYK version of RGBColorSeparate."

(*CMYKDominantAdd*)
CMYKDominantAdd::usage = "CMYK version of RGBDominantAdd."

(*ColoredWhiteNoise*)
ColoredWhiteNoise::usage = "ColoredWhiteNoise is a function whose action is whitening the image matrix, thus yielding to a signal corresponding to a white noise emission
function of the input image" 

(*ComicbookEffect*)
ComicbookEffect::usage = "Automatic comic artwork rendering thanks to edge detection and image compositions."

(*CompoundEye*)
CompoundEye::usage = "A thumbnail of arbitrary dimensions is created and a number of copies of it are juxtaposed to form a new image"

(*ContourTrace*)
ContourTrace::usage = "Contour trace is a famous Photoshop effect, and with Photoshop you just apply a filter (as usual). Here we first obtain the contour of the image 
thanks to morphological processing, and then map every single black point to a color."

(*EdgePlay*)
EdgePlay::usage = "It turns out that the difference between gradient and laplacian filtering is great also in artistic effects, and that keeping on multiplying a gradient
image with itself produces a stencil-like image. Modulating the number of times each channel has its gradient elevated determines a great range of effects (check the code)."

(*EffettoPentito*)
EffettoPentito::usage = "Classical way to obfuscate a face through pixelization."

(*FlipperScreen*)
FlipperScreen::usage = "This algo reproduces the images given by classic flipper screen, ie orange dotted."

(*FourierPhase*)
FourierPhase::usage = "Same implementation as FourierSpectrum, returning the phase matrix "

(*FourierSpectrum*)
FourierSpectrum::usage = "This function computes the spectrum of the discrete 'Fourier' transform applied to an image; the result, which is an array, is finally plotted. As usual, the plot
brings in its center the middlepoint. Recall that, according to 'Fourier' analysis, we're doing nothing but porting the original discrete signal (the argument image) into the frequency domain;
the term 'Fourier' spectrum denotes the module of the polar form of the transform."

(*GaussianScale*)
GaussianScale::usage = "Generate a gaussian kernel, turn it to an image through ArrayPlot and multiply the input with it:
ImageCrop[ColorNegate[ArrayPlot[GaussianMatrix[{200,80}],ImageSize\[Rule]{2449,2449},Frame\[Rule]False]]]
ImageResize[,{2222,2222}]
ImageMultiply[ImageResize[,{2222,2222}],%] UNIMPLEMENTED"

(*GrayscaleClassicAnaglyph*) 
GrayscaleClassicAnaglyph::usage =
"This algorithm produces one of those old fashioned redgreen 3d pictures, by superimposing a red and a cyan image out of phase; best viewed with red (left) and cyan (right) 3d glasses, go get some! 
Warning: the coefficients for the overlay coordinates as well as the one needed to cutoff the monochromatic part of the picture were stimated with common sense from a sample picture: quality may lack..."

(*HSVToRGB*)
HSVToRGB::usage =
"HSVToRGB[img] converts a hue-saturation-value triple to a red-green-blue one."

(*Kaleidoscope*)
Kaleidoscope::usage = "One of the many possible ways to obtain a kaleidoscopic image, through multiplications of the four basic reflections."

(*ImageLineIntegralConvolution*)
ImageLineIntegralConvolution::usage = "LIC is a way of depicting continuous field lines given a seed grid; in our case, the seed is an input image, maybe white noise or similar,
on which the effect of drawing the field becomes visible. The result is something affine, but rather different both theoretically and in practice, to geometric nonlinear 
distortion effects. This is a possible method for obtaining a motion-blur effect too."

(*ImagePadSize*)
ImagePadSize::usage =
"ImagePadSize[img1,img2,col] applies ImagePad with border color col on img1 in order to make it match the size of img2. This can be useful if you want to compose an image
from a set of subimages via ImageAssemble, as components on the same row have to share the same size, but at the same time appearance must be preserved and so simply resizing
can't be accepted."

(*ImagePlanes*)
ImagePlanes::usage = 
"ImagePlanes[img] gives eight bitplanes of img; img must be a byte-valued image having one or three channels (generalizations are trivial, check the code)."

(*ImageShear*)
ImageShear::usage =
"Regular shear transform, taking advantage of the structures Mathematica can readily apply to the Raster primitive (which we can obtain directly from ImageData)."

(*ImageSignal*)
ImageSignal::usage =
"ImageSignal[img,n,type] plots the signal function representing the image in the spatial domain MxN obtained via interpolation over the image data. Parameter n can assume values
1 (image is converted to grayscale) or 3 (the number of channels is not modified) and parameter type defines the type in which image data is retrivied through ImageData. If the
input image has more than three channels is converted to grayscale (as we cannot plot in 4D or more!)."

(*Interlace*)
Interlace::usage =
"Interlace[img] returns two copies of img where one has even lines set to zero, the other the odd ones. Interlacing frames into two subimages like these, called fields,
is a common technique in video processing. The computation is made by multiplying the image with two proper pattern masks and the inversion is given by summing the fields.
It can easily be adapted into a line screen effect (and it's already some kind of)."

(*LineOverlay*)
LineOverlay::usage = 
"Line overlay is a popart-like effect where the background is a fixed colors and the foreground different shades of another. Nice effect if added to a gradient or laplacian
image."

(*MeanSquareError*)
MeanSquareError::usage =
"MeanSquareError[imgx,imgy] gives the MSE between images imgx and imgy of the same kind obviously (useful to quantify the difference between two differently quantized copies of an image)."

(*Moroney*)
Moroney::usage = "Moroney[img] is an implementation of a well-known algorithm to enhance mixed-exposition images, originally described in Local color correction using nonlinear masking.
Parameters are automatically set."

(*NeonGlow*)
NeonGlow::usage = 
"NeonGlow returns a particular edge image resembling e neon light glowing in the dark."

(*PeakSignalToNoiseRatio*)
PeakSignalToNoiseRatio::usage =
"PeakSignalToNoiseRatio[imgx,imgy] is an image statistic based on MeanSquareError."

(*Pixellate*)
Pixellate::usage = 
"Pixellate[img,level] performs a 'blocketization' on the image induced by consecutive subsampling and upsampling interpolation; the image is resized by a scale given by level
and then restored to its original dimension."

(*Pointillize*)
Pointillize::usage = "Classical and simple but still appealing algorithm."

(*RandomGradient*)
RandomGradient::usage =
"Applies a randomly picked gradient among the ones available on the platform."

(*RCColorSeparate*)
RCColorSeparate::usage =
"RGBColorSeparate[img] assumes img to be an rgb one and it returns a list of the two chromatic components."

(*RGColorSeparate*)
RGColorSeparate::usage =
"RGBColorSeparate[img] assumes img to be an rgb one and it returns a list of the two chromatic components."

(*RGBColorSeparate*)
RGBColorSeparate::usage =
"RGBColorSeparate[img] assumes img to be an rgb one and it returns a list of the three chromatic components."

(*RGBDominantAdd*)
RGBDominantAdd::usage = "Modifies the image in order to obtain an x-ish or a x^-1-ish one, where x is obviously one of R, G, B; the specified channel is scaled by the given factor and the other two by the inverse of it. Zero is a nonsense value and negative values result in a black image;
notice that y>1 actually shades in the specified channel whereas y<1 exalts the complementary tone (ie (R, 0.5)->cyan hue)."

(*RGBThreshold*)
RGBThreshold::usage = "Given a center color rgb and a radius {rr,rg,rb} every color inside the bubble is unmodified whereas outsiders are sent back gray. This is the known single hue-grayscale effect."

(*RGBThreshold2*)
RGBThreshold2::usage = "Just like the previous one but instead of returning a gray we return an argument color "

(*RGBToHSV*)
RGBToHSV::usage =
"RGBToHSV[rgb] converts the triple from the rgb to the hsv space."

(*RGBToRC*)
RGBToRC::usage =
"RGBToRC[rgb] converts an rgb triple to the rc color space. No proper inverse is known "

(*RGBToRG*)
RGBToRG::usage =
"RGBToRG[rgb] converts an rgb triple to the rb color space. No proper inverse is known "

(*RGBToYCbCr*)
RGBToYCbCr::usage =
"RGBToYCbCr[rgb] converts the triple from the rgb to the YCbCr, a space where Y, Cb and Cr are the luminance and the two chrominance components respectively."

(*RGBToYDbDr*)
RGBToYDbDr::usage =
"YDbDrToRGB[rgb] converts a ydbdr triple to a ydbdr one."

(*RGBToYIQ*)
RGBToYIQ::usage =
"RGBToYIQ[rgb] converts a rgb triple to a yiq one."

(*RGBToYUV*)
RGBToYUV::usage =
"RGBToYUV[rgb] converts a rgb triple to a yuv one."

(*RMSContrast*)
RMSContrast::usage =
"RMSContrast[img] gives the contrast of img computed through the root mean square statistic; img must be a three-channel one (generalizations are trivial, check the code)."

(*RYBColorSeparate*)
RYBColorSeparate::usage =
"RYBColorSeparate assumes img to be an rgb one and it returns a list of the three chromatic components."

(*ShadeGradient*)
ShadeGradient::usage =
"ShadeGradient[img,name,di] returns a collection of shaded images according to the colors of a gradient range (namely, one of the many supported by 7).
As gradients are implemented as continuous sets, we need to sample n times with distance di over them in order to build the list."

(*ShadeIndexedColorTable*)
ShadeIndexedColorTable::usage =
"ShadeIndexedColorTable[img,num] returns a list of x images and each of them is a shaded version of img according to the x colors of ColorData[num,ColorList], obtained through
ShadeSingleColor."

(*ShadePhysicalGradient*)
ShadePhysicalGradient::usage =
"ShadePhysicalGradient[img,name,di] returns a collection of shaded images according to the colors of a phyisical spectrum (namely, the only three supported by 7).
As gradients are implemented as continuous sets over the [0,1] range, we need to sample n times with distance di over them in order to build the list."

(*ShadeSingleColor*)
ShadeSingleColor::usage = 
"ShadeSingleColor[img,{r,g,b}] applies to img a color gradient running from white to black centered in {r,g,b}. The effect is related to the XYZColorSeparate family."

(*SpraycanEffect*)
SpraycanEffect::usage = "This algorithm emulates a graffiti-like spraycan artwork."

(*Swirl*)
Swirl::usage = "This is a simple-and-plain swirl geometric effect: pixels are rotated by a matrix function of their distance from a center point."

(*ValueNoise*)
ValueNoise::usage = 
"Value noise is a cool noise effect where the grid is filled via a free interpolation on a few points. The result is paint-drops-like white patches on black (also, the algorithm
yields a lot of modifications in its implementation so modifying the result is pretty straightforward."

(*YCbCrToRGB*)
YCbCrToRGB::usage =
"YCbCrToRGB[ycbcr] converts a ycbcr triple to a rgb one."

(*YDbDrToRGB*)
YDbDrToRGB::usage =
"YDbDrToRGB[ydbdr] converts a ydbdr triple to a rgb one."

(*YIQToRGB*)
YIQToRGB::usage =
"YIQToRGB[yiq] converts a yiq triple to a rgb one."

(*YUVToRGB*)
YUVToRGB::usage =
"YUVToRGB[yuv] converts a yuv triple to a rgb one."

(*WorleyNoise*)
WorleyNoise::usage = "This algorithm produces a noise image useful to render rocks and water (read the code for a brief explanation)."


Begin["`Private`"]

BayerCFA[img_]:=
If[
ImageQ[img],
Module[{
rgbimgs=ColorSeparate[ColorConvert[img,"RGB"]],
m=ImageDimensions[img][[1]],
n=ImageDimensions[img][[2]]
},
Module[{
r=rgbimgs[[1]],
g=rgbimgs[[2]],
b=rgbimgs[[3]],
(*creation of pattern masks for zeroing according to bayer cfa...*)
rmask=Image[Array[BCFAR,{n,m}]],
gmask=Image[Array[BCFAG,{n,m}]],
bmask=Image[Array[BCFAB,{n,m}]]
},
Module[{
(*through multiplications*)
cfar=ImageMultiply[r,rmask],
cfag=ImageMultiply[g,gmask],
cfab=ImageMultiply[b,bmask]
},
ColorCombine[{cfar,cfag,cfab}]
]
]
],
Print["Not an image."]
]
BCFAR[i_,j_]:=Boole[EvenQ[i]&&EvenQ[j]] (*auxiliary function*)
BCFAG[i_,j_]:=Boole[OddQ[i]&&EvenQ[j] || (EvenQ[i]&&OddQ[j])] (*auxiliary function*)
BCFAB[i_,j_]:=Boole[OddQ[i]&&OddQ[j]] (*auxiliary function*)

Blueprint[img_,xlines_,ylines_]:=
If[
ImageQ[img],
Module[{
m=ImageDimensions[img][[1]],
n=ImageDimensions[img][[2]],
(*this pseudo blue-cyan is the color I found they use when images are transformed with this effect*)
base=ImageApply[{0.06089637744436507,0.31443169890538314,0.5716200506293385}&,img],
(*the edge image will be overlaid on the monochrome image base*)
edge=ColorConvert[GradientFilter[img,1],"Grayscale"]
},
Module[{
(*it turns out that sharpening also determines a nice "waterdrop " noise effect*)
sedge=Sharpen[edge,2]
},
Module[{
(*still waiting for the grid*)
nogrid=ImageAdd[base,sedge],
(*let's build the grid as a matrix; the grain of the grid is constant (good for the average image dimension)*)
grid=Table[If[Mod[i,Round[n/ylines]]==0||Mod[j,Round[m/xlines]]==0,{1,1,1},{0,0,0}],{i,n},{j,m}]
},
(*final result*)
Image[ImageData[nogrid]+grid]
]
]
],
Print["Error: not an image "]
]

CMYKColorSeparate[img_] :=
 If[ImageChannels[img] == 4,
  {
   ImageApply[{#[[1]], #[[2]] 0, #[[3]] 0, #[[4]] 0} &, img],
   ImageApply[{#[[1]] 0, #[[2]], #[[3]] 0, #[[4]] 0} &, img],
   ImageApply[{#[[1]] 0, #[[2]] 0, #[[3]], #[[4]] 0} &, img],
   ImageApply[{#[[1]] 0, #[[2]] 0, #[[3]] 0, #[[4]]} &, img]
   },
  Print["Error: incompatible picture; check the number of channels "]
  ]

CMYKDominantAdd[img_, chan_, val_] :=
 If[
  ImageChannels[img] == 
    4 && (chan == "C " || chan == "M " || chan == "Y " || chan == "K "),
  Switch[chan,
   "C ", ImageApply[{#[[1]] val, #[[2]] (1/val), #[[3]] (1/
         val), #[[4]] (1/val)} &, img],
   "M ", ImageApply[{#[[1]] (1/val), #[[2]] val, #[[3]] (1/
         val), #[[4]] (1/val)} &, img],
   "Y ", ImageApply[{#[[1]] (1/val), #[[2]] (1/
         val), #[[3]] val, #[[4]] (1/val)} &, img],
   "K ", ImageApply[{#[[1]] (1/val), #[[2]] (1/val), #[[3]] (1/
         val), #[[4]] val} &, img]
   ],
  Print["Error: incompatible picture or wrong params; check the \
number of channels or the image and the channel and value \
parameters"]
  ]

ColoredWhiteNoise[img_]:=
(*
this whitening function is the one proposed on en.wiki;
the overall algorithm is quite slow, check out the dimensions of img; the image is considered a realization of a random vector "X=(X_ 1,X_ 2,X_ 3)" 
where each component is a multivariate random variable "X_i=(x_i1,...x_i (m*n))", [0,1]-valued;
the two conditions on the mean vector and correlation matrix are respected with enough precision (MatrixPower[d,-.5].e.({r,g,b}-{mr,mg,mb}) is the random
vector whose mean has to be 0 and autocorrelation matrix the variance diagonal)
*)
If[
ImageQ[img],
Module[{
m=ImageDimensions[img][[1]],
rgbimg=ColorConvert[img,"RGB"]
(*in order to avoid distinguishing cases and follow what said before, though this might not be the rigorous approach*)
},
Module[{
r=Flatten[ImageData[ColorSeparate[img][[1]]]],
g=Flatten[ImageData[ColorSeparate[img][[2]]]],
b=Flatten[ImageData[ColorSeparate[img][[3]]]]
},
Module[{
mr=Table[Mean[r],{i,1,Length[r]}],
mg=Table[Mean[g],{i,1,Length[r]}],
mb=Table[Mean[b],{i,1,Length[r]}],
k=Covariance[Transpose[{r,g,b}]] 
(*covariance matrix of the image seen as an mxn vector on [0,1]^3*)
},
Module[ {
d=DiagonalMatrix[Eigenvalues[k]],
(*diagonal matrix of k's eigenvalues*)
e=Eigenvectors[k]
(*transposed orthogonal matrix of k's eigenvectors*)
},
Module[{
w=Rescale[MatrixPower[d,-.5].e.({r,g,b}-{mr,mg,mb})]
(*raw white image matrix scaled in [0,1]*)
},
Module[{
wr1=w[[1]],
wg1=w[[2]],
wb1=w[[3]]
},
Module[{
wr=Partition[wr1,m],
wg=Partition[wg1,m],
wb=Partition[wb1,m]
(*re-create the proper component images*)
},
ColorCombine[
{
Image[wr],
Image[wg],
Image[wb]
}
]
(*
to Grayscale because Raster always creates a 3-channel image; in the process the matrix gets rotated pi clockwise, so we gotta rotate it back
*)
]
]
]
]
]
]
],
Print["Error: not an image "]
]

ComicbookEffect[img_,intensity_]:=
(*The first part of the algorithm is identical to SpraycanEffect and in the rest of the code the crucial difference is adopting the Laplacian instead of the Gradient)*) 
(*Intensity levels to be considered effectively satisfying range from 10 up to 110*)
If[
ImageQ[img],
Module[{
cq=ColorQuantize[img,5] 
(*this contributes to a "pseudo " reduction of the overall number of colors, through patching*)
},
Module[{
im=ImageMultiply[img,cq]
},
Module[{
gf=GaussianFilter[im,3]
},
Module[{
s=Sharpen[gf,3]
},
Module[{
gf2=GaussianFilter[GaussianFilter[s,3],3]
},
Module[{
grf=GradientFilter[gf2,1]
},
Module[{
cc=ColorConvert[grf,"Grayscale"]
},
Module[{
ia=ImageAdd[ImageAdd[gf2,cc],cc]
},
Module[{
lf=LaplacianFilter[gf2,1] 
},
Module[{
cn=ColorNegate[lf]
},
Module[{
mask=Nest[ImageMultiply[cn,#]&,cn,Round[intensity]]

},
Lighter[ImageMultiply[ia,ColorConvert[mask,"Grayscale"]],intensity/3]
]
]
]
]
]
]
]
]
]
]
],
Print["Error: not an image "]
]

CompoundEye[img_,unitsize_,shape_,radius_,bground_]:=
(*This code is a neat example of how in mathematica we can use symbolic data in extravagant ways, obtaining a fly-like eye of an arbitrary shape is an easy task thanks to automatic generation an substitutions. Input is basically unchecked, try shape={80,60}, radius=10 and bground={r,g,b} and you shouldn't encounter any freezes/infinite computations*)
If[
ImageQ[img],
Module[{
m,
tnail=ImageResize[img,unitsize],
blank=ColorConvert[ImageResize[Image[{{bground}}],unitsize],"RGB"]
(*ImageAssemble requires argument images to be defined on the same colspace*)
},
Switch[
shape,
"Disk ",m=DiskMatrix[radius],
"Box ",m=BoxMatrix[radius],
"Cross ",m=CrossMatrix[radius],
"Diamond ",m=DiamondMatrix[radius],
_,Print["Error: valid shapes are Disk, Box, Cross and Diamond "]
];
Module[{
final=ReplaceAll[ReplaceAll[m,  1->tnail],0->blank]
},
ImageAssemble[final]
]
],
Print["Error: not an image "]
]

ContourTrace[img_,level0_,col_]:=
(*You can try out with values lower than 1 in the index argument of Table and see what's going on*)
If[
ImageQ[img],
Module[{
cn=Fold[ImageMultiply,ColorNegate[MorphologicalPerimeter[img,level0]],Table[ColorNegate[MorphologicalPerimeter[img,i]],{i,.1,1,.1}]]
},
Switch[
col,
"RGB",
Module[{
rgb={{1,0,0},{0,1,0},{0,0,1}}
},
ImageApply[If[#==0,RandomChoice[rgb],{1,1,1}]&,cn]
],
"CMYK",
Module[{
cmyk={{0,1,1},{1,0,1},{1,1,0},{0,0,0}}
},
ImageApply[If[#==0,RandomChoice[cmyk],{1,1,1}]&,cn]
],
"random",
ImageApply[If[#==0,{Random[],Random[],Random[]},{1,1,1}]&,cn],
_,
Print["Error: valid color specifications are RGB, CMYK and random"]
]
],
Print["Error: not an image"]
]


EdgePlay[img_,intensities_,filter_]:=
(*
The intensities define how bold the stencil/spraycan we're working with, and affect the dominants of the output image too. Grayscale inputs don't give greyscale outputs unless the intensities are all equal (infact the final image is returned as an rgb, regardless of the original input). 
Some cool inputs are:,
-EdgePlay[grayscaleimg,{100,100,100},2] to have something halfway between charcoal and stencil,
-EdgePlay[grayscaleimg,{200,200,200},1] silverplate ({100,200,200} lead),
-EdgePlay[img,{RandomInteger[{1,1000}],RandomInteger[{1,1000}],RandomInteger[{1,1000}]},1], dope!,           -EdgePlay[ColorSeparate[EdgePlay[ColorConvert[img,"Grayscale"],{RandomInteger[{1,1000}],RandomInteger[{1,1000}],RandomInteger[{1,1000}]},1]][[2]],{Random[],Random[],Random[]}]
 ImageSubtract[ColorSeparate[img][[2]],%], weird things,
-Shade ex. 1 to white, invert colors, shade randomly
*)
If[
ImageQ[img],
Module[{
nimg=ColorConvert[img,"RGB"]
},
Module[{
ir=intensities[[1]],
ig=intensities[[2]],
ib=intensities[[3]],
r=ColorSeparate[nimg][[1]],
g=ColorSeparate[nimg][[2]],
b=ColorSeparate[nimg][[3]]
},
Module[{
b1=GaussianFilter[r,6],
b2=GaussianFilter[g,6],
b3=GaussianFilter[b,6],
l1,
l2,
l3
},
If[
filter==1,
l1=ColorNegate[GradientFilter[b1,1]];
l2=ColorNegate[GradientFilter[b2,1]];
l3=ColorNegate[GradientFilter[b3,1]],
l1=ColorNegate[LaplacianFilter[b1,1]];
l2=ColorNegate[LaplacianFilter[b2,1]];
l3=ColorNegate[LaplacianFilter[b3,1]],
Print["Error: 1 to play with gradient, 2 to play with laplacian "]
];
Module[{
(*intensities can be big numbers, things go smooth even with numbers around 1000!*)
s1=Nest[ImageMultiply[l1,#]&,l1,ir],
s2=Nest[ImageMultiply[l2,#]&,l2,ig],
s3=Nest[ImageMultiply[l3,#]&,l3,ib]
},
ColorCombine[{s1,s2,s3}]
]
]
]
],
Print["Error: not an image "]
]

EffettoPentito[img_,fromrow_,torow_,fromcol_,tocol_,level_]:=
(*things are easy thanks to the pixelization principle; remember: always use Image!*)
If[
ImageQ[img],
Module[{
m=ImageDimensions[img][[1]],
n=ImageDimensions[img][[2]],
futureimg=ImageData[img]
},
Module[{
img2=ImageResize[ImageResize[img,{Round[m/Abs[level]],Round[n/Abs[level]]}],{m,n}]
},
(*if we're over the area to be obfuscated, we pick the blocky pixels*)
For[i=fromcol,i<tocol+1,i++,
For[j=fromrow,j<torow+1,j++,
futureimg[[j,i]]=ImageData[img2][[j,i]]
]
];
Image[futureimg]
]
],
Print["Error: not an image "]
]

FlipperScreen[img_,finesse_,rgb_]:=
(*nice effect with finesse 175 on orange color*)
If[
ImageQ[img],
Module[{
mask=Graphics[{RGBColor[rgb],Point[Flatten[Table[{i,j},{i,1,Abs[finesse]},{j,1,Abs[finesse]}],1]]},Background->Black,ImageSize->Max[ImageDimensions[img]]]
},
ImageMultiply[ColorConvert[img,"Grayscale"],mask]
],
Print["Error: not an image"]
]

FourierPhase[img_]:=
(*Dal codice per dare l' array spettro cambia veramente poco: qua abbiamo da applicare la fase al posto che l' assoluto, non utilizziamo il padding: pena, immagine poco dettagliata) e trascuriamo anche la log (non siamo nel regno dell' oscurita' col sole in mezzo come per lo spettro)*)
If[
	ImageQ[img],
	Module[{
		tutta=Arg[Fourier[ImageData[ColorConvert[img,"Grayscale"]],FourierParameters->{-1,1}]]
		},
		Module[{
	m=Dimensions[tutta][[1]],(*righe*)
	n=Dimensions[tutta][[2]]    (*colonne*)
	},
	Module[{
	altosx=Take[tutta,{1,Floor[m/2]},{1,Floor[n/2]}],
	altodx=Take[tutta,{1,Floor[m/2]},{Floor[n/2],n}],
	bassosx=Take[tutta,{Floor[m/2],m},{1,Floor[n/2]}],
	bassodx=Take[tutta,{Floor[m/2],m},{Floor[n/2],n}]
	},
	ImageCrop[ArrayPlot[Log[Join[Join[bassodx,bassosx,2],Join[altodx,altosx,2]]],Frame->False]]
]
]
],
Print["Error: not an image "]
]

FourierSpectrum[img_]:=
(*L' idea e' applicare la dft (o la fft? non so sotto cosa applichi poi il sistema) ad un' immagine; i parametri sono specifici per il data processing generico. 
Lavoriamo su greyscale perche' l' idea di potenza del segnale 'colorato' non e' persa, ragionevolmente il grigio e' una media ed anche perche' via ArrayPlot e' l' unico modo che ho trovato :L
L' immagine data e' prima di essere passata bordata di bianco (windowing function) per uno spettro visivamente migliore; tali bordi sono infine croppati*)

If[
	ImageQ[img],
	Module[{
		tutta=Abs[Fourier[ImageData[ImagePad[ColorConvert[img,"Grayscale"],256,White]],FourierParameters->{-1,1}]]
		},
		Module[{
	m=Dimensions[tutta][[1]],(*righe*)
	n=Dimensions[tutta][[2]]    (*colonne*)
	},
	(*Siccome vogliamo avere il 'punto di luce' al centro, dobbiamo ritagliare e ricomporre i quattro quadranti della matrice...*)
	Module[{
	altosx=Take[tutta,{1,Floor[m/2]},{1,Floor[n/2]}],
	altodx=Take[tutta,{1,Floor[m/2]},{Floor[n/2],n}],
	bassosx=Take[tutta,{Floor[m/2],m},{1,Floor[n/2]}],
	bassodx=Take[tutta,{Floor[m/2],m},{Floor[n/2],n}]
	},
	ImageCrop[ArrayPlot[Log[Join[Join[bassodx,bassosx,2],Join[altodx,altosx,2]]],Frame->False]]
]
]
],
Print["Error: not an image "]
]

GaussianScale[img_,s_,r_]:=Print["UNIMPLEMENTED"]

GrayscaleClassicAnaglyph[img_]:=
If[
ImageChannels[img]==3,
Module[{
imgr=ImageApply[{#[[1]],#[[2]]0,#[[3]]0}&,ColorConvert[ColorConvert[img,"Grayscale"],"RGB"]],
imgc=ImageApply[{#[[1]]0,#[[2]],#[[3]]}&,ColorConvert[ColorConvert[img,"Grayscale"],"RGB"]]
},
Module[{
a=ImageCompose[imgr,{imgc,0.5},{Ceiling[ImageDimensions[img][[1]]*183/343]-12,Ceiling[ImageDimensions[img][[2]]*255/512]}]
},
ImageTake[a,ImageDimensions[a][[2]],{17*ImageDimensions[a][[1]]/343,ImageDimensions[a][[1]]}]
]
],
Print["Error: incompatible picture; check the number of channels "]
]

HSVToRGB[hsv_] :=
 Module[{
   h = hsv[[1]],
   s = hsv[[2]],
   v = hsv[[3]]
   },
  If[
   h < 0 || h >= 360 || s < 0 || s > 1 || v < 0 || v > 1,
   Print["Error: out of range; correct intervals are: 0<=h<360, \
0<=s,v<=1"],
   Module[{
     hi = Mod[Floor[h/60], 6],
     f = h/60 - Floor[h/60],
     p = v (1 - s)},
    Module[{
      q = v (1 - f s),
      t = v (1 - (1 - f) s)
      },
     Switch[hi,
      0, {v, t, p},
      1, {q, v, p},
      2, {p, v, t},
      3, {p, q, v},
      4, {t, p, v},
      5, {v, p, q}
      ]
     ]
    ]
   ]
  ]

Kaleidoscope[img_]:=
(*This algorithm doesn't preserve the input chromatic content as various copies of the image are overlaid, and you might want to lighten the output*)
If[
ImageQ[img],
Module[{
m=ImageDimensions[img][[1]],
n=ImageDimensions[img][[2]],
a=ImageReflect[img,Top->Bottom],
b=ImageReflect[img,Top->Left],
c=ImageReflect[img,Bottom->Left],
d=ImageReflect[img,Left->Right]
},
Module[{
im=ImageMultiply[ImageMultiply[ImageMultiply[a,b],c],d]
},
Module[{
(*the multiplications decrease the pixel values; this amount is sometimes unsufficient*)
l=Lighter[im,20]
},
If[
(*different ways depending on being the image a horizontal or a vertical one*)
m>=n,
Module[{
(*flipping the image makes it easier to crop it*)
ir=ImageRotate[l,90Degree]
},
(*cropping is necessary because the input images we joined are of two different dimensions, mxn an nxm*)
Module[{
it=ImageTake[ir,{Round[(m-n)/2],Round[m-Round[(m-n)/2]]}]
},
ImageRotate[it,-90Degree]
]
],
Module[{
(*the inverse of the cut for the m-dominant case*)
it=ImageTake[l,{Round[(n-m)/2],Round[n-Round[(n-m)/2]]}]
},
ImageRotate[it,-90Degree]
]
]
]
]
],
Print["Error: not an image"]
]

ImageLineIntegralConvolution[img_,field_]:=
(*The vector fields I propose are some of those you can find in the manual pages of LineIntegralConvolutionPlot and VectorPlot*)
If[
ImageQ[img],
Module[{
m=ImageDimensions[img][[1]],
n=ImageDimensions[img][[2]],
r=(x^2+y^2)^.5,
(*division by zero is properly handed by the system,a valid output image is given as well*)
s=ArcTan[y/x]
},
Switch[
field,
"Flow ",ImageCrop[
LineIntegralConvolutionPlot[{{-1-x^2+y,1+x-y^2},img},{x,-3,3},{y,-3,3},Frame->False,RasterSize->{m,n},AspectRatio->n/m]
],
"Sinusoidal ",ImageCrop[
LineIntegralConvolutionPlot[{{Sin[x],Sin[y]},img},{x,0,2Pi},{y,0,2Pi},Frame->False,RasterSize->{m,n},AspectRatio->n/m]
],
"Swirl ",ImageCrop[
LineIntegralConvolutionPlot[{{-y,x},img},{x,-3,3},{y,-3,3},Frame->False,RasterSize->{m,n},AspectRatio->n/m]
],
"Diamond ",ImageCrop[
LineIntegralConvolutionPlot[{{Sin[s] Cos[r],Cos[s]Sin[r]},img},{x,0,2Pi},{y,0,2Pi},Frame->False,RasterSize->{m,n},AspectRatio->n/m]
],
"Handkerchief ",ImageCrop[
LineIntegralConvolutionPlot[{{r Sin[s+r],r Cos[s-r]},img},{x,0,2Pi},{y,0,2Pi},Frame->False,RasterSize->{m,n},AspectRatio->n/m]
],
"Disc ",ImageCrop[
LineIntegralConvolutionPlot[{{s/Pi Sin[Pi r],s/Pi Cos[Pi r]},img},{x,0,2Pi},{y,0,2Pi},Frame->False,RasterSize->{m,n},AspectRatio->n/m]
],
"River ",ImageCrop[
LineIntegralConvolutionPlot[{{x+y,x-y},img},{x,1,100},{y,1,100},Frame->False,RasterSize->{m,n},AspectRatio->n/m]
],
"Twirl ",ImageCrop[
LineIntegralConvolutionPlot[{{y,-Sin[x]},img},{x,-4,4},{y,-4,4},Frame->False,RasterSize->{m,n},AspectRatio->n/m]
],
"Nodal ",ImageCrop[
LineIntegralConvolutionPlot[{{x,2y},img},{x,-1,1},{y,-1,1},Frame->False,RasterSize->{m,n},AspectRatio->n/m]
],
"Degenerate ",ImageCrop[
LineIntegralConvolutionPlot[{{x+y,y},img},{x,-1,1},{y,-1,1},Frame->False,RasterSize->{m,n},AspectRatio->n/m]
],
"Saddle ",ImageCrop[
LineIntegralConvolutionPlot[{{x,-2y},img},{x,-1,1},{y,-1,1},Frame->False,RasterSize->{m,n},AspectRatio->n/m]
],
_,Print["Error: valid vector field names are: Flow, Sinusoidal, Swirl, Diamond, Handkerchief, Disc, River, Twirl, Nodal, Degenerate and Saddle."]
]
],
Print["Error: not an image "]
]

ImagePadSize[img1_,img2_,col_]:=
If[
(*img1 is the image to be padded accordingly to img2 dimensions*)
ImageQ[img1] && ImageQ[img2],
Module[{
m1=ImageDimensions[img1][[1]],
n1=ImageDimensions[img1][[2]],
m2=ImageDimensions[img2][[1]],
n2=ImageDimensions[img2][[2]]
},
Module[{
img1p=ImagePad[
img1,{{Round[(m2-m1)/2],Round[(m2-m1)/2]},{Round[(n2-n1)/2],Round[(n2-n1)/2]}},col
]},
(*if we're lucky, sizes match; if we aren't, the difference isn't that much anyways!*)
If[ImageDimensions[img1p]!= ImageDimensions[img2],ImageResize[img1p,{m2,n2}],img1p]
]
],
Print["Error: not an image "]
]

ImagePlanes[img_] :=
If[
 ImageChannels[img] == 3,
 Module[{
   a = ColorSeparate[img]
   }, {
   ImageApply[If[# <= 0.5,#*0,#*0 + 1] &, a[[1]]],
   ImageApply[If[# <= 0.25,#*0,#*0 + 1] &, a[[1]]],
   ImageApply[If[# <= 0.125,#*0,#*0 + 1] &, a[[1]]],
   ImageApply[If[# <= 0.0625,#*0,#*0 + 1] &, a[[1]]],
   ImageApply[If[# <= 0.03125,#*0,#*0 + 1] &, a[[1]]],
   ImageApply[If[# <= 0.015625,#*0,#*0 + 1] &, a[[1]]],
   ImageApply[If[# <= 0.0078125,#*0,#*0 + 1] &, a[[1]]],
   ImageApply[If[# <= 0.00390625,#*0,#*0 + 1] &, a[[1]]],
   ImageApply[If[# <= 0.5,#*0,#*0 + 1] &, a[[2]]],
   ImageApply[If[# <= 0.25,#*0,#*0 + 1] &, a[[2]]],
   ImageApply[If[# <= 0.125,#*0,#*0 + 1] &, a[[2]]],
   ImageApply[If[# <= 0.0625,#*0,#*0 + 1] &, a[[2]]],
   ImageApply[If[# <= 0.03125,#*0,#*0 + 1] &, a[[2]]],
   ImageApply[If[# <= 0.015625,#*0,#*0 + 1] &, a[[2]]],
   ImageApply[If[# <= 0.0078125,#*0,#*0 + 1] &, a[[2]]],
   ImageApply[If[# <= 0.00390625,#*0,#*0 + 1] &, a[[2]]],
   ImageApply[If[# <= 0.5,#*0,#*0 + 1] &, a[[3]]],
   ImageApply[If[# <= 0.25,#*0,#*0 + 1] &, a[[3]]],
   ImageApply[If[# <= 0.125,#*0,#*0 + 1] &, a[[3]]],
   ImageApply[If[# <= 0.0625,#*0,#*0 + 1] &, a[[3]]],
   ImageApply[If[# <= 0.03125,#*0,#*0 + 1] &, a[[3]]],
   ImageApply[If[# <= 0.015625,#*0,#*0 + 1] &, a[[3]]],
   ImageApply[If[# <= 0.0078125,#*0,#*0 + 1] &, a[[3]]],
   ImageApply[If[# <= 0.00390625,#*0,#*0 + 1] &, a[[3]]]
   }
  ],
 If[
  ImageChannels[img] == 1,
  {
   ImageApply[If[# <= 0.5,#*0,#*0 + 1] &, img],
   ImageApply[If[# <= 0.25,#*0,#*0 + 1] &, img],
   ImageApply[If[# <= 0.125,#*0,#*0 + 1] &, img],
   ImageApply[If[# <= 0.0625,#*0,#*0 + 1] &, img],
   ImageApply[If[# <= 0.03125,#*0,#*0 + 1] &, img],
   ImageApply[If[# <= 0.015625,#*0,#*0 + 1] &, img],
   ImageApply[If[# <= 0.0078125,#*0,#*0 + 1] &, img],
   ImageApply[If[# <= 0.00390625,#*0,#*0 + 1] &, img]
   },
  Print["Error: incompatible picture; check the number of channels "]
  ]
 ]

ImageShear[img_,shear_,bg_]:=
(*It's strange that no examples in the refpage of GeometricTransformation show how to do this; 
things are not exaggerately tricky, we just have to pass through Raster and Graphics.
Check out ShearingTransform for information about the three needed parameters, an angle and two vectors, you have to provide in the parameter shear*)
If[
ImageQ[img],
Module[{
m=ImageDimensions[img][[1]],
n=ImageDimensions[img][[2]],
data=Reverse[ImageData[img]](*to avoid unwanted reflections along the second diagonal*)
},
Module[{
transform=GeometricTransformation[Raster[data],ShearingTransform[shear[[1]],shear[[2]],shear[[3]]]
]
},
(*rendering*)
Graphics[transform,Background->RGBColor[bg]]
]
],
Print["Error: not an image "]
]

ImageSignal[img_,n_,type_]:=
If[
ImageQ[img],
Module[{
a=ImageDimensions[img][[1]],
b=ImageDimensions[img][[2]],
},
If[
n==1 || ImageChannels[img]>3,
Module[{
gsimg=ColorConvert[img,"Grayscale"]
},
Module[{
data=Flatten[ImageData[gsimg,type]]
},
Module[{
f=ListInterpolation[{data}]
},
(*per tipo dati:Real a e b andrebbero cambiati, 
ma 1 e' ideale per gli altri tre*)
Plot3D[f[x,y],{x,1,a},{y,1,b}] 
]
]
],
Module[{
data=Flatten[ImageData[img,type]]
},
Module[{
f=ListInterpolation[{data}]
},
Plot3D[f[x,y],{x,1,a},{y,1,b}]
]
]
]
],
Print["Error: not an image "]
]

Interlace[img_]:=
If[
ImageQ[img] && ImageChannels[img]==3,
Module[{
m=ImageDimensions[img][[1]],
n=ImageDimensions[img][[2]]
},
Module[{
(*attenzione, array chiede prima l' altezza n poi la larghezza m*)
data1=Array[WhiteBlackPattern,{n,m}],
data2=Array[BlackWhitePattern,{n,m}]
},
Module[{
(*senza complicarci la vita con raster e graphics*)
img1=Image[data1],
img2=Image[data2]
},
{ImageMultiply[img,img1],ImageMultiply[img,img2]}
]
]
],
Print["Error: incompatible picture; check the number of channels "]
]
(*Mathematica pare non supportare argomenti di tipo "any term "*)
WhiteBlackPattern[i_,j_]:=Boole[OddQ[i]] (*auxiliary function*)
BlackWhitePattern[i_,j_]:=Boole[Not[OddQ[i]]] (*auxiliary function*)

LineOverlay[img_,fg_,bg_,intensity_]:=
If[
ImageQ[img],
Module[{
m=ImageDimensions[img][[1]],
n=ImageDimensions[img][[2]],
t=Table[ColorQuantize[GaussianFilter[img,5],i],{i,2,15}]
},
Module[{
t2=ShadeSingleColor[#,fg]& /@ t 
(*this first shading is fundamental in the logic of the algorithm*)
},
Module[{
f=Fold[ImageAdd,First[t2],Rest[t2]]
(*core intuition*)
},
Module[{
ssc=ShadeSingleColor[ShadeSingleColor[ColorNegate[f],fg],fg]
(*two shadings give proper color*)
},
Module[{
ia=ImageApply[If [#=={1,1,1},{0,0,0},#]&,
ImageApply[If[#=={0,0,0},bg,#]&,ColorConvert[Binarize[ssc],"RGB"]]
(*the mask has the requested background and is blank everywhere else*)
]
},
Sharpen[MeanFilter[ImageAdd[ssc,ia],2],Round[intensity]]
(*the mean filter helps us in killing most of the quantization artifacts into nice color blowouts and the sharpening yields to flame-like effects*)
]
]
]
]
],
Print["Error: not an image"]
]

MeanSquareError[imgx_,imgy_]:=
(*L'idea e' quella di fare l' mse sui tre canali separatamente e sommare*)
If[ImageQ[imgx]&&ImageQ[imgy]&&ImageDimensions[imgx]==ImageDimensions[imgy]&&ImageType[imgx]==ImageType[imgy]&&ImageChannels[imgx]==ImageChannels[imgy],
(*applichiamo il prodotto tra le dimensioni dell' immagine:*)
Module[{n=Apply[(#1 #2)&,ImageDimensions[imgx]]},
(*sommiamo le tre componenti mse, una per ogni canale*)
Apply[(#1[[1]]+#2[[2]]+#3[[3]])&,
(*applichiamo la definizione: differenza tra valori pixel*)
(1/n) Apply[((#1[[1]]-#2[[1]])^2)&,{ImageData[imgx],ImageData[imgy]}];
(1/n) Apply[((#1[[2]]-#2[[2]])^2)&,{ImageData[imgx],ImageData[imgy]}];
(1/n) Apply[((#1[[3]]-#2[[3]])^2)&,{ImageData[imgx],ImageData[imgy]}]]],
Print["Error: incompatible picture couple; check they share the same dimensions, type, number of channels "]]

Moroney[img_]:=
(*Implementation of the tone filter discussed in Local color correction using nonlinear masking*)
(*Portare in matlab*)
Module[{
hsb=ColorConvert[img,"HSB"],
m=ImageDimensions[img][[1]],
n=ImageDimensions[img][[2]]
},
Module[{
h=ColorSeparate[hsb][[1]],
s=ColorSeparate[hsb][[2]],
b=ColorSeparate[hsb][[3]],
std=.02Sqrt[m^2+n^2] (*suggested by Durand*)
},
Module[{
r=Ceiling[6std],(*standard cutoff*)
bn=ColorNegate[b],
(*the algorithm is designed for a byte-valued image*)
b8=Image[Rescale[ImageData[b],{0,1},{0,255}],"Byte"]
},
Module[{
bnm=GaussianFilter[bn, {r,std}],
newb8,
newhsb
},
(*that's neat! setting ImageData manually is a no-no (this ain't matlab folks!)*)
newb8=Table[
255(PixelValue[b8,{i,j},"Byte"]/255)^2^((128-PixelValue[bnm,{i,j}])/128),{j,1,n},{i,1,m}
];
ColorCombine[{
(*we need to rotate and stuff as we create the new value channel manually*)
h,s,ImageReflect[ImageRotate[ImageRotate[Image[Rescale[newb8]]]],Left]
},"HSB"]
]
]
]
]

NeonGlow[img_,rgb_,intensity_]:=
(*The intensity parameter is the number of times the base image is added to increasingly-gaussianed masks; the higher it is, the larger the neonish emission area around the objects will be*)
If[
ImageQ[img],
Module[{
gf=GradientFilter[img,3]
},
Module[{
cc=ColorConvert[gf,"Grayscale"]
},
Module[{
ssc=ShadeSingleColor[cc,rgb]
},
Module[{
masks=Table[GaussianFilter[ssc,10i],{i,1,Abs[intensity]}]
},
Fold[ImageAdd,ssc,masks]
]
]
]
],
Print["Error: not an image"]
]

PeakSignalToNoiseRatio[imgx_, imgy_] := 10 Log[10, (1/MeanSquareError[imgx, imgy])]

Pixellate[img_,level_]:=
If[
ImageQ[img],
Module[{
m=ImageDimensions[img][[1]],
n=ImageDimensions[img][[2]]
},
ImageResize[ImageResize[img,{Round[m/Abs[level]],Round[n/Abs[level]]}],{m,n}]
],
Print["Error: not an image "]
]

Pointillize[img_,finesse_]:=
(*In order to avoid "tiling pointillization", you can set yourself the ImageSize of the mask to some greater multiple couple (say, 4; you can use Abs[frac] too). With 2, the highest finesse is around 270*)
If[
ImageQ[img],
Module[{
m=ImageDimensions[img][[1]],
n=ImageDimensions[img][[2]]
},
Module[{
	mask=Graphics[{RGBColor[{1,1,1}],Point[Flatten[Table[{i,j},{i,1,Abs[finesse]},{j,1,Abs[finesse]}],1]]},Background->Black,ImageSize->{3m,3n}]
},
Module[{
(*For proper multiplication*)
mask2=Image[ImageData[mask]]
},
ImageMultiply[img,mask2]
]
]
],
Print["Error: not an image"]
]

RandomGradient[img_]:=
 If[
  ImageQ[img],
  Module[{cdg=ColorData["Gradients"]},
   Module[{g=cdg[[RandomInteger[{1,Length[cdg]}]]]},
    ImageApply[List@@ColorData[ToString[g],#]&,ColorConvert[img,"Grayscale"]]]],
  Print["Error: not an image "]]

RCColorSeparate[img_] :=
If[
ImageChannels[img] == 3,
Module[{
rc=ImageApply[RGBToRC,img],
k=ImageMultiply[ColorSeparate[img][[1]],0]
},
{
ImageApply[{#[[1]], #[[2]] 0, #[[3]] 0} &, img],
ColorCombine[{k,ColorSeparate [rc][[2]],ColorSeparate [rc][[2]]}]
}
],
  Print["Error: incompatible picture; check the number of channels "]
  ]

RGColorSeparate[img_] :=
(*dummy from RGBColorSeparate*)
 If[ImageChannels[img] == 3,
  {
   ImageApply[{#[[1]], #[[2]] 0, #[[3]] 0} &, img],
   ImageApply[{#[[1]] 0, #[[2]], #[[3]] 0} &, img]
   },
  Print["Error: incompatible picture; check the number of channels "]
  ]

RGBColorSeparate[img_] :=
 If[ImageChannels[img] == 3,
  {
   ImageApply[{#[[1]], #[[2]] 0, #[[3]] 0} &, img],
   ImageApply[{#[[1]] 0, #[[2]], #[[3]] 0} &, img],
   ImageApply[{#[[1]] 0, #[[2]] 0, #[[3]]} &, img]
   },
  Print["Error: incompatible picture; check the number of channels "]
  ]

RGBDominantAdd[img_, chan_, val_] :=
 If[
  ImageChannels[img] == 
    3 && (chan == "R" || chan == "G" || chan == "B"),
  Switch[chan,
   "R", ImageApply[{#[[1]] val, #[[2]] (1/val), #[[3]] (1/val)} &, 
    img],
   "G", ImageApply[{#[[1]] val, #[[2]] (1/val), #[[3]] (1/val)} &, 
    img],
   "B", ImageApply[{#[[1]] (1/val), #[[2]] (1/val), #[[3]] val} &, img]
   ],
  Print["Error: incompatible picture or wrong params; check the \
number of channels or the image and the channel and value \
parameters"]
  ]

RGBThreshold[img_,c_,rr_,rg_,rb_]:=
If[
ImageQ[img],
ImageApply[rgbthreshold[#,c,Abs[rr],Abs[rg],Abs[rb]]&,ColorConvert[img,"RGB"]],
Print["Error: not an image "]
]
rgbthreshold[rgb_,c_,rr_,rg_,rb_]:=
(*Consider a bubble of radius r=(rr,rg,rb) in the rgb space centered in color c; if the argument 
color rgb is inside the bubble, we "save " it, otherwise we ground it to gray*)
If[
  c[[1]]-rr<=rgb[[1]]<=c[[1]]+rr&&
c[[2]]-rg<=rgb[[2]]<=c[[2]]+rg&&
c[[3]]-rb<=rgb[[3]]<=c[[3]]+rb,
rgb,
{
.3rgb[[1]]+.6rgb[[2]]+.1rgb[[3]],
.3rgb[[1]]+.6rgb[[2]]+.1rgb[[3]],
.3rgb[[1]]+.6rgb[[2]]+.1rgb[[3]]
}
(*this is how Mathematica converts an rgb triplet into a greyscale one*)
]

RGBThreshold2[img_,to_,c_,rr_,rg_,rb_]:=
If[
ImageQ[img],
ImageApply[rgbthreshold2[#,to,c,Abs[rr],Abs[rg],Abs[rb]]&,ColorConvert[img,"RGB"]],
Print["Error: not an image "]
]
rgbthreshold2[rgb_,to_,c_,rr_,rg_,rb_]:=
If[
  c[[1]]-rr<=rgb[[1]]<=c[[1]]+rr&&
c[[2]]-rg<=rgb[[2]]<=c[[2]]+rg&&
c[[3]]-rb<=rgb[[3]]<=c[[3]]+rb,
to,
{
.3rgb[[1]]+.6rgb[[2]]+.1rgb[[3]],
.3rgb[[1]]+.6rgb[[2]]+.1rgb[[3]],
.3rgb[[1]]+.6rgb[[2]]+.1rgb[[3]]
}
]

RGBToHSV[rgb_] :=
 Module[{
   r = rgb[[1]],
   g = rgb[[2]],
   b = rgb[[3]]
   },
  If[
   r < 0 || r > 1 || g < 0 || g > 1 || b < 0 || b > 1,
   Print["Error: out of range; correct intervals are: 0<=r,g,b<=1"],
   Module[{
     max = Max[rgb],
     min = Min[rgb],
     h,
     s,
     v
     },
    Switch[max,
     min, h = 0,
     r, h = Mod[60 ((g - b)/(max - min)) + 360, 360],
     g, h = 60 ((b - r)/(max - min)) + 120,
     b, h = 60 ((r - g)/(max - min)) + 240
     ];
    If[
     max == 0,
     s = 0,
     s = 1 - min/max
     ];
    v = max;
    Return[{h, s, v}]
    ]
   ]
  ]

RGBToRC[rgb_]:=
Module[{
   r = rgb[[1]],
   g = rgb[[2]],
   b = rgb[[3]]
   },
  If[
   r < 0 || r > 1 || g < 0 || g > 1 || b < 0 || b > 1, 
   Print["Error: out of range; correct intervals are: 0<=r,g,b<=1"],
   Return[
	(*weighted average because green is more important than blue; blue is even ignored in many implementations*)
    {r,(g+0.5b)/1.5,(g+0.5b)/1.5} 
    ]
   ]
  ]

RGBToRG[rgb_]:=
Module[{
   r = rgb[[1]],
   g = rgb[[2]],
   b = rgb[[3]]
   },
  If[
   r < 0 || r > 1 || g < 0 || g > 1 || b < 0 || b > 1, 
   Print["Error: out of range; correct intervals are: 0<=r,g,b<=1"],
   Return[
    {r,g,0}
    ]
   ]
  ]

RGBToYCbCr[rgb_] :=
 Module[{
   r = rgb[[1]],
   g = rgb[[2]],
   b = rgb[[3]]
   },
  If[
   r < 0 || r > 1 || g < 0 || g > 1 || b < 0 || b > 1, 
   Print["Error: out of range; correct intervals are: 0<=r,g,b<=1"],
   Return[
    {0.2989 r + 0.5866 g + 0.1145 b,
     -0.1687 r - 0.3312 g + 0.5b,
     0.5 r - 0.4183 g - 0.0816 b}
    ]
   ]
  ]

RGBToYDbDr[rgb_] :=
 Module[{
   r = rgb[[1]],
   g = rgb[[2]],
   b = rgb[[3]]
   },
  If[
   r < 0 || r > 1 || g < 0 || g > 1 || b < 0 || b > 1, 
   Print["Error: out of range; correct intervals are: 0<=r,g,b<=1"],
   Return[
    {0.299 r + 0.587 g + 0.114 b,
     -0.45 r - 0.883 g + 1.333 b,
     -1.333 r + 1.116 g + 0.217 b}
    ]
   ]
  ]

RGBToYUV[rgb_] :=
 Module[{
   r = rgb[[1]],
   g = rgb[[2]],
   b = rgb[[3]]
   },
  If[
   r < 0 || r > 1 || g < 0 || g > 1 || b < 0 || b > 1, 
   Print["Error: out of range; correct intervals are: 0<=r,g,b<=1"],
   Return[
    {0.299 r + 0.587 g + 0.114 b,
     -0.14713 r - 0.28886 g - 0.436 b,
     0.615 r - 0.51499 g -0.10001 b}
    ]
   ]
  ]

RGBToYIQ[rgb_] :=
 Module[{
   r = rgb[[1]],
   g = rgb[[2]],
   b = rgb[[3]]
   },
  If[
   r < 0 || r > 1 || g < 0 || g > 1 || b < 0 || b > 1, 
   Print["Error: out of range; correct intervals are: 0<=r,g,b<=1"],
   Return[
    {0.299 r + 0.587 g + 0.114 b,
     0.595716 r - 0.274453 g - 0.321263 b,
     0.211456 r - 0.522591 g + 0.311135 b}
    ]
   ]
  ]

RMSContrast[img_] :=
 If[
  ImageChannels[img] == 3,
  Module[{
    x = Flatten[ImageData[ColorSeparate[img][[1]]]], 
    y = Flatten[ImageData[ColorSeparate[img][[2]]]], 
     z = Flatten[ImageData[ColorSeparate[img][[3]]]]
    },
   {StandardDeviation[x], StandardDeviation[y], StandardDeviation[z]}
   ],
  If[ImageChannels[img] == 1, 
   StandardDeviation[Flatten[ImageData[img]]], 
   Print["Error: incompatible picture; check the number of channels "]]]

RYBColorSeparate[img_] :=
 If[ImageChannels[img] == 3,
  {
   ImageApply[{#[[1]], #[[2]] 0, #[[3]] 0} &, img],
   ImageApply[{(#[[1]]0.5+#[[2]])/1.5,(#[[1]]0.5+#[[2]])/1.5,#[[3]]0}&,img],(*media pesata a favore del verde*)
   ImageApply[{#[[1]] 0, #[[2]] 0, #[[3]]} &, img]
   },
  Print["Error: incompatible picture; check the number of channels "]
  ]

ShadeGradient[img_,name_,dx_]:=
If[
ImageQ[img], 
Module[{
images={},
samples=Table[i,{i,0,1,dx}]
},
Module[{
huesList=ColorData[name]/@samples
},
For[
i=1,i<=Length[huesList],i++,
AppendTo[
images,
ShadeSingleColor[img,ToExpression[StringReplace[ToString[huesList[[i]]],{"RGBColor["->"{","]"->"}"}]]
]
]
]
];
images
],
Print["Error: not an image "]
]

ShadeIndexedColorTable[img_,number_]:=
If[
ImageQ[img],
Module[{
huesList=ColorData[number,"ColorList"], (*listing all the RGB color values for the requested collection*)
images={}
},
For[
(*dal di dentro:
-l' i-esimo elemento di huesList e' un' espressione, la convertiamo a stringa
 -in questa stringa, sostituiamo per farla diventare una lista
 -pero' e' formalmente ancora una stringa, e per essere argomento di ShadeSingleColor dev' essere invece un' espressione
 -ricordiamoci AppendTo e non Append seno' non va un cazzo dinamicamente
*)
i=1,i<=Length[huesList],i++,AppendTo[images,ShadeSingleColor[img,ToExpression[StringReplace[ToString[huesList[[i]]],{"RGBColor["->"{","]"->"}"}]]]]
];
Select[images,ImageQ] (*infact some calls unexpectedly give subfailures resuling in a corrupted list*)
],
Print["Error: not an image "]
]

ShadePhysicalGradient[img_,name_,dx_]:=
(*This code is pretty static, the three possible (atm) cases are treated explicitly.
Pay attention to the sampling step dx: small values will surely cause infinite computations :L
I treat the physical ones differently as they range in their own intervals rather than [0,1] as other gradients*)
If[
ImageQ[img],
Switch[name,(*Switch is neater than If*)

"VisibleSpectrum", 
Module[{
images={},
samples=Table[i,{i,380,750,dx}]
},
Module[{
huesList=ColorData[name]/@samples
},
For[
i=1,i<=Length[huesList],i++,
AppendTo[
images,
ShadeSingleColor[img,ToExpression[StringReplace[ToString[huesList[[i]]],{"RGBColor["->"{","]"->"}"}]]]
]
]
];
images
],

"BlackBodySpectrum",
Module[{
images={},
samples=Table[i,{i,1000,10000,dx}]
},
Module[{
huesList=ColorData[name]/@samples
},
For[
i=1,i<=Length[huesList],i++,
AppendTo[
images,
ShadeSingleColor[img,ToExpression[StringReplace[ToString[huesList[[i]]],{"RGBColor["->"{","]"->"}"}]]]
]
]
];
images
],

"HypsometricTints",
Module[{
images={},
samples=Table[i,{i,-6000,6000,dx}]
},
Module[{
huesList=ColorData[name]/@samples
},
For[
i=1,i<=Length[huesList],i++,
AppendTo[
images,
ShadeSingleColor[img,ToExpression[StringReplace[ToString[huesList[[i]]],{"RGBColor["->"{","]"->"}"}]]]
]
]
];
images
],

_,
Print["Unknown physical gradient: known gradients are VisibleSpectrum, BlackBodySpectrum, HypsometricTints"]

],
Print["Error: not an image "]
]

ShadeSingleColor[img_,rgb_]:=
If[
ImageQ[img]&&(0<=rgb[[1]]<= 1&&0<=rgb[[3]]<= 1&&0<=rgb[[3]]<= 1),
Module[{
colorfunction=Interpolation[{{0,{0,0,0}},{0.5,rgb},{1,{1,1,1}}}](*lame interpolation, but it proved to be neat*)
},
ImageApply[colorfunction[#]&,ColorConvert[img,"Grayscale"]]
],
Print["Not an image. Or out of range rgb values (0<=r,g,b<=1 is lecit)"]
]

SpraycanEffect[img_,intensity_]:=
(*This algorithm transforms the input image into a graffiti-like artwork, thanks to color quantization and derivative-of-gaussian filters. The intensity represents the weight of the gradient mask obtained recursively (see EdgePlay)*) 
If[
ImageQ[img],
Module[{
cq=ColorQuantize[img,5] 
(*this contributes to a "pseudo " reduction of the overall number of colors, through patching*)
},
Module[{
im=ImageMultiply[img,cq]
},
Module[{
gf=GaussianFilter[im,3]
},
Module[{
s=Sharpen[gf,3]
},
Module[{
gf2=GaussianFilter[GaussianFilter[s,3],3]
},
Module[{
grf=GradientFilter[gf2,1]
},
Module[{
cc=ColorConvert[grf,"Grayscale"]
},
Module[{
cn=ColorNegate[cc]
},
Module[{
mask=Nest[ImageMultiply[cn,#]&,cn,Round[intensity]]
(*try low values if you notice you're losing details (or eccessive artifacts are visible); anyway, the window of a sufficiently good spray effect is [10,35]*)
},
Lighter[ImageMultiply[gf2,mask],intensity/10]
(*gradient filtering heavy edges result into loss of brightness, here's the remedy*)
]
]
]
]
]
]
]
]
],
Print["Error: not an image "]
]

Swirl[img_,factor_]:=
(*This algorithm was copied "as is" from the pages of a princeton course about Java programming. Things can surely be done better (more parameters, interpolation etc)*)
If[
ImageQ[img],
Module[{
m=ImageDimensions[img][[1]],
n=ImageDimensions[img][[2]]
},
Module[{
i0=Round[.5(m-1)],
j0=Round[.5(n-1)],
nimg=Table[{0,0,0},{i,1,m},{j,1,n}],
di,dj,r,a,rt,ti,tj
},
For[i=1,i<m,i++
For[j=1,j<n,j++,
di=i-i0;
dj=j-j0;
r=Sqrt[di^2+dj^2];
a=Pi/factor r;
rt=RotationTransform[a,{i0,j0}];
ti=Round[di Cos[a]-dj Sin[a]+i0];
tj=Round[di Sin[a]+dj Cos[a]+j0];
If[(1<=ti<= m)&&(1<=tj<=n),nimg[[i]][[j]]=ImageData[img][[tj,ti]]]
(*fucking imagedata wants reversed indexes!*)
]
];
ImageRotate[Image[nimg],Right]
]
],
Print["Error: not an image"]
]

ValueNoise[m_,n_,p_]:=
(*The concept of value noise is to define some pixel values in the grid and find the empty ones interpolating. The algorithm is way faster than the one for Worley noise.*)
Module[{
grid=Table[0,{i,n},{j,m}],
points=Table[{{RandomInteger[{1,n}],RandomInteger[{1,m}]},Random[]},{i,p}]
},
Module[{
(*
 -interpolation can be one or another, polynomial effectively works with random points
-quasi binary images
-the amount of implicit interpolation over extrapolation through the polynomial is due to how much close to the extremes of the grid the random sample points actually are
*)
(*InterpolatingPolynomial doesn't like repeated abscissas...*)
ip=InterpolatingPolynomial[DeleteDuplicates[points,#1[[1]]==#2[[1]]&],{x,y}]
},
For[i=1,i<n,i++
For[j=1,j<m,j++,
(*evaluating the interpolant in every point of the grid*)
grid[[i]][[j]]=Rescale[ip/.{x->i,y->j},{-Infinity,+Infinity},{0,1}]
]
];
Image[grid]
]
]

YCbCrToRGB[ycbcr_] :=
 Module[{
   y = ycbcr[[1]],
   cb = ycbcr[[2]],
   cr = ycbcr[[3]]
   },
  If[
   y < 0 || y > 1 || cb < -0.5 || cb > 0.5 || cr < -0.5 || cr > 0.5, 
   Print["Error: out of range; correct intervals are: 0<=y<=1;-0.5<=cb,cr<=0.5"],
   Return[
    {y + 1.4022 cr,
     y - 0.3456 cb - 0.7145 cr,
     y + 1.7710 cb}
    ]
   ]
  ]

YIQToRGB[yiq_] :=
 Module[{
   y = yiq[[1]],
   i = yiq[[2]],
   q = yiq[[3]]
   },
  If[
   y < 0 || y > 1 || i < -0.5957 || i > 0.5957 || q < -0.5226 || q > 0.5226, 
   Print["Error: out of range; correct intervals are: 0<=y<=1;-0.5957<=i<=0.5957;-0.5226<=q<=0.5226"],
   Return[
    {y + 0.9563 i + 0.621 q,
     y - 0.2721 i - 0.6474 q,
     y - 1.107 i + 1.7046 q}
    ]
   ]
  ]

YDbDrToRGB[ydbdr_] :=
 Module[{
   y = ydbdr[[1]],
   db = ydbdr[[2]],
   dr = ydbdr[[3]]
   },
  If[
   y < 0 || y > 1 || db < -1.333 || db > 1.333 || dr < -1.333 || dr > 1.333, 
   Print["Error: out of range; correct intervals are: 0<=y<=1;-1.333<=db,dr<=1.333"],
   Return[
    {y + 0.000092303716148db + \[Minus] 0.525912630661865dr,
     y \[Minus] 0.129132898890509db + 0.267899328207599dr ,
     y 0.664679059978955db \[Minus] 0.000079202543533dr }
    ]
   ]
  ]

YUVToRGB[yuv_] :=
 Module[{
   y = yuv[[1]],
   u = yuv[[2]],
   v = yuv[[3]]
   },
  If[
   y < 0 || y > 1 || u < -0.436 || u > 0.436 || v < -0.615 || v > 0.615, 
   Print["Error: out of range; correct intervals are: 0<=y<=1;-0.436<=u<=0.436;-0.615<=v<=0.615"],
   Return[
    {y + 1.13983 v,
     y - 0.39465 u - 0.5806 v,
     y + 2.03211 u}
    ]
   ]
  ]

WorleyNoise[m_,n_,p_,k_]:=
(*This algorithm returns a greyscale image representing random worley noise, the ref paper is
"A cellular texture basis function " by S. Worley. The idea is:
-on an m by n grid, choose randomly p points
 -define a function on the grid, whose value is the distance between a pixel point and the
 k-closest of these randomly picked points (I choose euclidean distance)
 -natively render the grid through Image
The needed time dramatically becomes "unfeasible " as the required grid is bigger (I think optimizations are possible, especially in the order statistics employed)
Take a look at the paper for further indications and examples*)
If[
k<=p (*otherwise the function F_n is not well defined*),
Module[{
function=Table[0,{i,n},{j,m}]
},
Module[{
(*random tiepoints*)
features=Table[{RandomInteger[{1,n}],RandomInteger[{1,m}]},{i,p}],
(*an entry [i,j] is the list of the distances between grid[i,j] and all the random points*)
distances=Table[{},{i,n},{j,m}]
},
(*compute these distances for every single point*)
For[i=1,i<n+1,i++,
For[j=1,j<m+1,j++,
For[l=1,l<p+1,l++,
AppendTo[distances[[i]][[j]],N[EuclideanDistance[{i,j},features[[l]]]]]
]
]
];
(*take the k-th*)
For[i=1,i<n+1,i++,
For[j=1,j<m+1,j++,
function[[i]][[j]]=Sort[distances[[i]][[j]]][[k]]
]
];
(*rescale*)
Module[{
img=Rescale[function,{0,Max[function]},{0,1}]
},
Image[img]
]
]
],
Print["Error: the grade of the function cannot be higher than the number of feature points "]
]

End[ ]
Print["ready..."]
EndPackage[ ]

























































