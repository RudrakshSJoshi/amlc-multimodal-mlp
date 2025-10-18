# Official Solution for Amazon ML Challenge 2025 by team SPAM_LLMs (IIT ISM Dhanbad) (3rd in Public LB, 5th in Private LB)

## Team Members
- Manav Jain [Linkedin](https://www.linkedin.com/in/manav-jain-05a711255/)
- Karaka Prasanth Naidu [Linkedin](https://www.linkedin.com/in/prasanth-naidu-karaka-a7162019b/)
- Alok Raj [Linkedin](https://www.linkedin.com/in/loki-silvres/)
- Rudraksh Sachin Joshi [Linkedin](https://www.linkedin.com/in/rudraksh-sachin-joshi-75554b202/)

## Problem Statement
### Smart Product Pricing Challenge
In e-commerce, determining the optimal price point for products is crucial for marketplace success and customer satisfaction.
Your challenge is to develop an ML solution that analyzes product details and predict the price of the product. The relationship between product attributes and pricing is complex - with factors like brand, specifications, product quantity directly influence pricing. Your task is to build a model that can analyze these product details holistically and suggest an optimal price.

Data Description:

The dataset consists of the following columns:
1. sample_id: A unique identifier for the input sample
2. catalog_content: Text field containing title, product description and an Item Pack Quantity(IPQ) concatenated.
3. image_link: Public URL where the product image is available for download. Example link - https://m.media-amazon.com/images/I/71XfHPR36-L.jpg
4. price: Price of the product (Target variable - only available in training data)

# Dataset 

Link: [Kaggle Dataset]()

# Approach

TL;DR Use Embeddings for different modality with Modality wise networks and a final regressor network.

Boomers: Oh you didnt finetune big models, not novel enough....

Just using pretrained embeddings without finetuning and careful selection and training of subsequent layers was able to beat many teams who adopted fine-tuning based approaches and reach 3rd position in public LB and 5th position in private LB. Even teams above us were within small margin of SMAPE. Using Pretrained frozen encoders might seem less cool.
It turns out, LLMs which were distilled to smaller versions lead to suboptimial representation capability and the quality of embeddings decrease drastically. Finetuning them can only take so far. So bigger models sometimes beat finetuned smaller models. 

One common misconception with teams was to select a single modality like text or image without experimenting. Using Image Modality in the beginning itself helped us realize the importance. It's easy to think that image might not be important to predict price, but getting information signal from various modalities can help improve performance.

All the above points were in respect to competitive settings, and can be loosely extended to production settings, where getting a better score is very important.

## Evaluation Metric
Symmetric Mean Absolute Percentage Error (SMAPE)

$$
\text{SMAPE} = \frac{100\%}{n} \sum_{i=1}^{n} 
\frac{|y_i - \hat{y}_i|}{\frac{|y_i| + |\hat{y}_i|}{2}}
$$

- $$\( y_i \)$$: true value  
- $$\( \hat{y}_i \)$$: predicted value  
- Range: $$\( [0, 200] \)$$  
- Lower is better 
# Handling Dataset
<Insert Distribution Image>


- Right Skewed Dataset -> Apply log1p transformation
- Same Image + Different Catalogs and prices , Same Catalogue + Different Images -> Need for multimodal approach
- Strip content catalog -> Production Description, Bullet Points, Quantity, Unit
- Contains Inconsistent unit representation -> Map them into a consistent representation (Ex: grams, gm, g -> grams)
- Convert Numbers to words -> Helps with getting better embeddings (Ex: 100 -> One Hundred)
- Large Catalog content -> If Product Description is available use it only, else use top 5 bullet points

# Embedding Generation

## Models Used:

After Experimenting with multiple models, we used the following models in our final pipeline

1. Qwen-3-4B Text Only[HF](https://huggingface.co/Qwen/Qwen3-4B)
2. Siglip2 Giant - 2B - Text + Image [HF](https://huggingface.co/google/siglip2-giant-opt-patch16-256)
3. DinoV3 - 0.8B - Image [HF](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)

Some teams tried finetuning smaller versions of Qwen3, but 4B version produced good enough embeddings which performed well and gave good improvemance gap. Try out bigger models before finetuning smaller models.

# Modality Specific Networks
< Insert Architecture Image >

Embeddings generated from each model are in different space, and it is hard for a model to learn from these embeddings by simply concating them. 
Applying Modality Specific Transformations helped training easier and converge quickly. We believe this specific method helped improve the performance.

# Loss function

## Log-based MSE Loss

$$
\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} 
\left( \log(y_i + 1) - \hat{y}_i \right)^2
$$

where  
- $$\( y_i \)$$: true target value  
- $$\( \hat{y}_i \)$$: predicted value (already in log scale)  
- $$\( n \)$$: number of samples  








