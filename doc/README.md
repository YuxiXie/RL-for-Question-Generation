# Supplementary Analysis 

## Effects of Hyper-Parameters

#### ![](http://latex.codecogs.com/gif.latex?\\alpha_{flu},\alpha_{rel},\alpha_{ans})

* ***fluency***: $\mathcal{L}_{flu} = -(\mathcal{R}_{flu}(\hat{\mathcal{Y}}) - \alpha_{flu})\frac{1}{T}\sum_{t=1}^T\log{P_{QG}(\hat{y}_t|\mathcal{D}, \hat{\mathcal{Y}}_{<t})}$

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
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
</tr>
<tr>
    <td>&emsp;-15</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
</tr>
</table>

* ***relevance***: $\mathcal{L}_{rel} = -(\mathcal{R}_{rel}(\mathcal{D}, \hat{\mathcal{Y}}) - \alpha_{rel})\frac{1}{T}\sum_{t=1}^T\log{P_{QG}(\hat{y}_t|\mathcal{D}, \hat{\mathcal{Y}}_{<t})}$

<table>
<tr>
    <td>&emsp;Values</td>
    <td>&emsp;BLEU-1</td>
    <td>&emsp;BLEU-4</td>
    <td>&emsp;METEOR</td>
    <td>&emsp;ROUGE-L</td>
</tr>
<tr>
    <td>&emsp;log2 = log(1/(1 - 0.5))</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
</tr>
<tr>
    <td>&emsp;log(1/(1 - 0.4))</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
</tr>
<tr>
    <td>&emsp;log(1/(1 - 0.6))</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
</tr>
</table>

* ***answerability***: $\mathcal{L}_{ans} = -(\mathcal{R}_{ans}(\mathcal{D}, \hat{\mathcal{Y}}) - \alpha_{ans})\frac{1}{T}\sum_{t=1}^T\log{P_{QG}(\hat{y}_t|\mathcal{D}, \hat{\mathcal{Y}}_{<t})}$

<table>
<tr>
    <td>&emsp;Values</td>
    <td>&emsp;BLEU-1</td>
    <td>&emsp;BLEU-4</td>
    <td>&emsp;METEOR</td>
    <td>&emsp;ROUGE-L</td>
</tr>
<tr>
    <td>&emsp;log2 = log(1/(1 - 0.5))</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
</tr>
<tr>
    <td>&emsp;log(1/(1 - 0.4))</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
</tr>
<tr>
    <td>&emsp;log(1/(1 - 0.6))</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
</tr>
</table>

<!-- <tr><td colspan="2"><a href="#resources">4. Resources</a></td></tr> -->

#### $\gamma_{flu}$, $\gamma_{rel}$, $\gamma_{ans}$

* ***Global Loss***: $\mathcal{L} = \mathcal{L}_{base} + \gamma_{flu}\mathcal{L}_{flu} + \gamma_{rel}\mathcal{L}_{rel} + \gamma_{ans}\mathcal{L}_{ans}$

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
    <td>&emsp;1</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
</tr>
<tr>
    <td>&emsp;0.1</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
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
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
</tr>
<tr>
    <td>&emsp;0.5</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
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
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
</tr>
<tr>
    <td>&emsp;0.5</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
    <td>&emsp;</td>
</tr>
</table>