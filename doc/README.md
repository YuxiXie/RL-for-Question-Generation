# Supplementary Analysis 

## Effects of Hyper-Parameters

#### ![](http://latex.codecogs.com/gif.latex?\\alpha_{flu},\alpha_{rel},\alpha_{ans})

* ***fluency***
![](http://latex.codecogs.com/gif.latex?\\mathcal{L}_{flu}=-(\mathcal{R}_{flu}(\hat{\mathcal{Y}})-\alpha_{flu})\frac{1}{T}\sum_{t=1}^T\log{P_{QG}(\hat{y}_t|\mathcal{D},\hat{\mathcal{Y}}_{<t})})

<table>
<tr>
    <td>&emsp;Values</td>
    <td>&emsp;BLEU-1</td>
    <td>&emsp;BLEU-4</td>
    <td>&emsp;METEOR</td>
    <td>&emsp;ROUGE-L</td>
</tr>
<tr>
    <td>&emsp;-10</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
</tr>
<tr>
    <td>&emsp;-5</td>
    <td>&emsp;38.74</td>
    <td>&emsp;15.62</td>
    <td>&emsp;18.58</td>
    <td>&emsp;34.29</td>
</tr>
<tr>
    <td>&emsp;-15</td>
    <td>&emsp;38.58</td>
    <td>&emsp;15.46</td>
    <td>&emsp;18.22</td>
    <td>&emsp;33.90</td>
</tr>
</table>

* ***relevance***
![](http://latex.codecogs.com/gif.latex?\\mathcal{L}_{rel}=-(\mathcal{R}_{rel}(\mathcal{D},\hat{\mathcal{Y}})-\alpha_{rel})\frac{1}{T}\sum_{t=1}^T\log{P_{QG}(\hat{y}_t|\mathcal{D},\hat{\mathcal{Y}}_{<t})})

<table>
<tr>
    <td>&emsp;Values</td>
    <td>&emsp;BLEU-1</td>
    <td>&emsp;BLEU-4</td>
    <td>&emsp;METEOR</td>
    <td>&emsp;ROUGE-L</td>
</tr>
<tr>
    <td>&emsp;log(1/(1 - 0.5)) = log2</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
</tr>
<tr>
    <td>&emsp;log(1/(1 - 0.4))</td>
    <td>&emsp;36.11</td>
    <td>&emsp;14.68</td>
    <td>&emsp;20.73</td>
    <td>&emsp;35.50</td>
</tr>
<tr>
    <td>&emsp;log(1/(1 - 0.6))</td>
    <td>&emsp;36.17</td>
    <td>&emsp;14.69</td>
    <td>&emsp;20.57</td>
    <td>&emsp;35.46</td>
</tr>
</table>

* ***answerability***
![](http://latex.codecogs.com/gif.latex?\\mathcal{L}_{ans}=-(\mathcal{R}_{ans}(\mathcal{D},\hat{\mathcal{Y}})-\alpha_{ans})\frac{1}{T}\sum_{t=1}^T\log{P_{QG}(\hat{y}_t|\mathcal{D},\hat{\mathcal{Y}}_{<t})})

<table>
<tr>
    <td>&emsp;Values</td>
    <td>&emsp;BLEU-1</td>
    <td>&emsp;BLEU-4</td>
    <td>&emsp;METEOR</td>
    <td>&emsp;ROUGE-L</td>
</tr>
<tr>
    <td>&emsp;log(1/(1 - 0.5)) = log2</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
</tr>
<tr>
    <td>&emsp;log(1/(1 - 0.4))</td>
    <td>&emsp;34.22</td>
    <td>&emsp;13.82</td>
    <td>&emsp;20.69</td>
    <td>&emsp;35.01</td>
</tr>
<tr>
    <td>&emsp;log(1/(1 - 0.6))</td>
    <td>&emsp;34.28</td>
    <td>&emsp;13.81</td>
    <td>&emsp;20.60</td>
    <td>&emsp;34.94</td>
</tr>
</table>

<!-- <tr><td colspan="2"><a href="#resources">4. Resources</a></td></tr> -->

#### ![](http://latex.codecogs.com/gif.latex?\\gamma_{flu}$,$\gamma_{rel}$,$\gamma_{ans})

* ***Global Loss***
![](http://latex.codecogs.com/gif.latex?\\mathcal{L}=\mathcal{L}_{base}+\gamma_{flu}\mathcal{L}_{flu}+\gamma_{rel}\mathcal{L}_{rel}+\gamma_{ans}\mathcal{L}_{ans})

<table>
<tr>
    <td>&emsp;Fluency</td>
    <td>&emsp;BLEU-1</td>
    <td>&emsp;BLEU-4</td>
    <td>&emsp;METEOR</td>
    <td>&emsp;ROUGE-L</td>
</tr>
<tr>
    <td>&emsp;0.2</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
</tr>
<tr>
    <td>&emsp;0.5</td>
    <td>&emsp;37.75</td>
    <td>&emsp;15.33</td>
    <td>&emsp;19.59</td>
    <td>&emsp;35.30</td>
</tr>
<tr>
    <td>&emsp;2</td>
    <td>&emsp;38.43</td>
    <td>&emsp;15.49</td>
    <td>&emsp;19.06</td>
    <td>&emsp;35.00</td>
</tr>
</table>

<table>
<tr>
    <td>&emsp;Relevance</td>
    <td>&emsp;BLEU-1</td>
    <td>&emsp;BLEU-4</td>
    <td>&emsp;METEOR</td>
    <td>&emsp;ROUGE-L</td>
</tr>
<tr>
    <td>&emsp;1</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
</tr>
<tr>
    <td>&emsp;2</td>
    <td>&emsp;37.17</td>
    <td>&emsp;15.06</td>
    <td>&emsp;20.16</td>
    <td>&emsp;35.46</td>
</tr>
<tr>
    <td>&emsp;0.5</td>
    <td>&emsp;37.60</td>
    <td>&emsp;15.16</td>
    <td>&emsp;19.54</td>
    <td>&emsp;35.24</td>
</tr>
</table>

<table>
<tr>
    <td>&emsp;Answerability</td>
    <td>&emsp;BLEU-1</td>
    <td>&emsp;BLEU-4</td>
    <td>&emsp;METEOR</td>
    <td>&emsp;ROUGE-L</td>
</tr>
<tr>
    <td>&emsp;1</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
</tr>
<tr>
    <td>&emsp;2</td>
    <td>&emsp;36.42</td>
    <td>&emsp;14.74</td>
    <td>&emsp;20.14</td>
    <td>&emsp;35.33</td>
</tr>
<tr>
    <td>&emsp;0.5</td>
    <td>&emsp;37.45</td>
    <td>&emsp;15.29</td>
    <td>&emsp;20.16</td>
    <td>&emsp;35.51</td>
</tr>
</table>