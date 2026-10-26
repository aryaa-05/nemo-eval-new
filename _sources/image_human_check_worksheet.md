# Image-processing human-check worksheet

Rules: open the cited source (Young, Gerbrands and van Vliet, Fundamentals of Image Processing, Version 2.3, 2007). For each item confirm: (1) the section/page exists; (2) the excerpt appears verbatim; (3) exactly the keyed option is supported; (4) the other three options are not supported. Fill PASS or FAIL, then tell me which IDs passed so the `human_checked` flags can be set.

Rows: 20 of 100. At least 20 are required for --final.

## 1. ip_003  (Quantization)

**Question:** Representing the amplitude at a coordinate as an integer value with L different gray levels is called what?

- A. Sampling
- B. Quantization **(KEY)**
- C. Interpolation
- D. Convolution

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 2, page 3

**Excerpt:** The process of representing the amplitude of the 2D signal at a given coordinate as an integer value with L different gray levels is usually referred to as amplitude quantization or simply quantization.

**Review flag:** False

**PASS/FAIL:** 

---

## 2. ip_010  (Convolution theorem)

**Question:** Convolution in the spatial domain is equivalent to what operation in the Fourier domain?

- A. Addition
- B. Differentiation
- C. Multiplication **(KEY)**
- D. Subsampling

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 3.4, page 10

**Excerpt:** In words, convolution in the spatial domain is equivalent to multiplication in the Fourier (frequency) domain and vice-versa.

**Review flag:** False

**PASS/FAIL:** 

---

## 3. ip_015  (DFT periodicity)

**Question:** The discrete-space Fourier transform A(Ω,Ψ) is periodic in both variables with what period?

- A. π in Ω only
- B. 2π in Ω only
- C. π in both variables
- D. 2π in both variables **(KEY)**

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 3.4, page 9

**Excerpt:** The Fourier transform in discrete space, A(Ω,Ψ), is periodic in both Ω and Ψ. Both periods are 2π.

**Review flag:** False

**PASS/FAIL:** 

---

## 4. ip_020  (Hankel transform)

**Question:** Which function appears in the Hankel transform used for circularly symmetric signals?

- A. A sinc function
- B. The zero-order Bessel function **(KEY)**
- C. A Gaussian function
- D. A rectangular boxcar

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 3.4.2, page 12

**Excerpt:** Jo(•) is a Bessel function of the first kind of order zero.

**Review flag:** False

**PASS/FAIL:** 

---

## 5. ip_023  (Density function)

**Question:** In the text, what is p(a) called in the expression p(a)Δa?

- A. The probability density function **(KEY)**
- B. The cumulative histogram
- C. The point spread function
- D. The optical transfer function

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 3.5.2, page 13

**Excerpt:** The probability that a brightness in a region falls between a and a+Δa, given the probability distribution function P(a), can be expressed as p(a)Δa where p(a) is the probability density function:

**Review flag:** False

**PASS/FAIL:** 

---

## 6. ip_027  (Standard deviation)

**Question:** The sample standard deviation sa is an estimate of what?

- A. The minimum brightness
- B. The standard deviation of the underlying brightness distribution **(KEY)**
- C. The median brightness
- D. The number of pixels in the region

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 3.5.4, page 17

**Excerpt:** The standard deviation, sa, is an estimate of σa of the underlying brightness probability distribution.

**Review flag:** False

**PASS/FAIL:** 

---

## 7. ip_030  (Mode)

**Question:** What is the mode of a brightness distribution?

- A. The average of the brightest pixels
- B. The most frequent brightness value **(KEY)**
- C. The difference between maximum and minimum
- D. The median of the lowest half

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 3.5.7, page 18

**Excerpt:** The mode of the distribution is the most frequent brightness value. There is no guarantee that a mode exists or that it is unique.

**Review flag:** False

**PASS/FAIL:** 

---

## 8. ip_038  (Weber-Fechner law)

**Question:** The Weber-Fechner law says the visual response is approximately proportional to what?

- A. The square of the intensity
- B. The logarithm of the intensity **(KEY)**
- C. The inverse of the intensity
- D. The intensity itself

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 4.1.2, page 22

**Excerpt:** to a good approximation, the visual response, R, is proportional to the logarithm of the intensity. This is known as the Weber-Fechner law.

**Review flag:** False

**PASS/FAIL:** 

---

## 9. ip_044  (Cutoff frequency)

**Question:** If a lens does not meet the circularly symmetric, aberration-free, diffraction-limited assumptions, what happens to its cutoff frequencies?

- A. They become higher than the diffraction-limited values
- B. They vanish completely
- C. They are lower than the diffraction-limited values **(KEY)**
- D. They become independent of the lens

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 5.1, page 29

**Excerpt:** If the lens does not meet one or more of these assumptions then it will still be bandlimited but at lower cutoff frequencies than those given in eq. (57).

**Review flag:** True

**PASS/FAIL:** 

---

## 10. ip_045  (Sampling aperture)

**Question:** When does the sampling aperture have only a marginal effect on the final signal?

- A. When the bandlimited and Nyquist conditions are satisfied **(KEY)**
- B. When the aperture is much wider than the sampling interval
- C. When the image contains no noise
- D. When the sensor has a binary output

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 5.1.1, page 30

**Excerpt:** The aperture p(x,y) described above will have only a marginal effect on the final signal if the two conditions eqs. (56) and (57) are satisfied.

**Review flag:** False

**PASS/FAIL:** 

---

## 11. ip_050  (Quantization noise)

**Question:** Where does quantization noise occur?

- A. In the optical lens system before the sensor
- B. In the ADC during amplitude quantization **(KEY)**
- C. In the display refresh circuit after conversion
- D. In the camera shutter mechanism during exposure

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 6.6, page 35

**Excerpt:** Quantization noise is inherent in the amplitude quantization process and occurs in the analog-to-digital converter, ADC.

**Review flag:** False

**PASS/FAIL:** 

---

## 12. ip_054  (Quantum efficiency)

**Question:** What does quantum efficiency describe?

- A. The number of pixels in a CCD row
- B. The average number of photoelectrons produced per photon at a wavelength **(KEY)**
- C. The ratio of readout rate to frame rate
- D. The maximum display refresh rate

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 7.2.2, page 37

**Excerpt:** The quantum efficiency refers to how many photoelectrons are produced—on the average—per photon at a given wavelength.

**Review flag:** False

**PASS/FAIL:** 

---

## 13. ip_056  (Shading)

**Question:** What does shading mean in the text?

- A. A constant input produces a constant digital image
- B. The camera always saturates at the maximum level
- C. The image histogram is perfectly equalized
- D. A constant physical input produces a nonconstant digital image **(KEY)**

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 7.4, page 39

**Excerpt:** By this we mean that if the physical input image a(x,y) = constant, then the digital version of the image will not be constant.

**Review flag:** False

**PASS/FAIL:** 

---

## 14. ip_064  (Histogram equalization function)

**Question:** For histogram equalization, what must be true of a suitable function f?

- A. It must be linear and decreasing
- B. It must be periodic with period B
- C. It must be differentiable with df/da greater than or equal to zero **(KEY)**
- D. It must map every input to the same output

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 9.1.2, page 46

**Excerpt:** From eq. (79) we see that suitable means that ƒ(•) is differentiable and that dƒ/da ≥ 0.

**Review flag:** False

**PASS/FAIL:** 

---

## 15. ip_067  (Arithmetic operations)

**Question:** Which arithmetic operation is defined as c = a / b?

- A. ADD
- B. SUB
- C. MUL
- D. DIV **(KEY)**

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 9.2.2, page 49

**Excerpt:** DIV c = a / b floating point

**Review flag:** False

**PASS/FAIL:** 

---

## 16. ip_068  (Convolution output)

**Question:** In convolution, what is the output pixel value?

- A. The weighted sum of input pixels within the window **(KEY)**
- B. The maximum input pixel within the window
- C. The median input pixel within the window
- D. The unweighted first pixel in the window

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 9.3, page 49

**Excerpt:** The output pixel value is the weighted sum of the input pixels within the window where the weights are the values of the filter assigned to every pixel of the window itself.

**Review flag:** False

**PASS/FAIL:** 

---

## 17. ip_078  (Gradient magnitude)

**Question:** Where does the magnitude gradient take on large values?

- A. Where there are strong edges in the image **(KEY)**
- B. Only in flat uniform regions
- C. Only at the image border
- D. Only where brightness is zero

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 9.5.1, page 65

**Excerpt:** The magnitude gradient takes on large values where there are strong edges in the image.

**Review flag:** False

**PASS/FAIL:** 

---

## 18. ip_082  (SDGD filter)

**Question:** How many partial derivatives does the SDGD filter use?

- A. Two
- B. Three
- C. Five **(KEY)**
- D. Nine

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 9.5.2, page 67

**Excerpt:** A filter that is especially useful in edge finding and object measurement is the Second-Derivative-in-the-Gradient-Direction (SDGD) filter. This filter uses five partial derivatives:

**Review flag:** False

**PASS/FAIL:** 

---

## 19. ip_093  (Shading model)

**Question:** In the shading model, how does illumination interact with the object?

- A. Additively
- B. Multiplicatively **(KEY)**
- C. Through a logarithm only
- D. Through thresholding

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 10.1.1, page 86

**Excerpt:** The illumination Iill(x,y) usually interacts in a multiplicative with the object a(x,y) to produce the image b(x,y):

**Review flag:** False

**PASS/FAIL:** 

---

## 20. ip_099  (Wiener filter)

**Question:** In what sense is the Wiener filter optimal for restoration in noise?

- A. Maximum contrast
- B. Minimum file size
- C. Minimum mean-square error **(KEY)**
- D. Zero latency

**Source:** Fundamentals of Image Processing, Young, Gerbrands and van Vliet, Version 2.3 (2007), Section 10.2.2, page 91

**Excerpt:** The word optimal is used here in the sense of minimum mean-square error ( mse).

**Review flag:** False

**PASS/FAIL:** 

---
