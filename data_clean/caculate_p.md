对每张图：
计算 reward 均值 
$$\mu_x
$$
计算 reward 标准差 
$$\sigma_x$$
计算中等难度分数
$$S_{\text{mid}}(x)=\exp\left(-\frac{(\mu_x-m_\mu)^2}{2(\tau_\mu s_\mu)^2+\epsilon}\right)$$
计算区分度分数
最稳的是用标准差或极差的归一化版本：
$$S_{\text{var}}(x)=1-\exp\left(-\frac{\sigma_x}{\tau_\sigma s_\sigma+\epsilon}\right)$$
或者更简单：
$$S_{\text{var}}(x)=\text{percentile}(\sigma_x)$$
合成价值
$$V(x)=S_{\text{mid}}(x)\cdot S_{\text{var}}(x)$$
归一化成采样概率
这个方案几乎就是你的需求的标准答案。
$$p(x)=\frac{V(x)}{\sum_{x'}V(x')}
$$