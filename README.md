# Predicting Creative Effectiveness For Advertising Campaigns

## 1. Overview

This project develops a multi-modal machine learning system that predicts creative effectiveness and
quantifies the emotional and aesthetic impact of ad creatives (images or videos).

By leveraging advances in representation learning , computer vision , and natural language understanding , the system aims to:

- Model the relationship between creative attributes and downstream engagement outcomes.
- Identify latent factors that drive ad performance.
- Deliver actionable, interpretable insights to guide creative optimization at scale.


## 2. Problem Statement

Creative effectiveness prediction is a multi-faceted, multimodal forecasting task that requires
learning from diverse signals — visual design, textual context, audience targeting, and campaign
metadata.

The goal is to estimate the probability distribution over performance metrics given creative and contextual features, while disentangling correlation from causal drivers.


## 3. Dataset: 
The dataset used in this project encompasses detailed information on various ad campaigns, specifically focusing on different attributes of the ads, their performance metrics, and associated metadata. Each row in the dataset represents an individual ad campaign instance with the following features:

- **campaign_item_id**: Unique identifier for each campaign item.
- **no_of_days**: Duration in days the ad campaign ran.
- **time**: The date on which the ad campaign was executed.
- **ext_service_id**: Identifier for the external service used for the ad campaign.
- **ext_service_name**: Name of the external service platform (e.g., Facebook Ads, Google Ads).
- **creative_id**: Identifier for the creative content used in the ad.
- **search_tags**: Tags associated with the ad for search optimization.
- **template_id**: Identifier for the ad template used.
- **landing_page**: URL of the landing page linked to the ad.
- **advertiser_id**: Unique identifier for the advertiser.
- **advertiser_name**: Name of the advertiser.
- **network_id**: Identifier for the ad network.
- **approved_budget**: Budget approved for the ad campaign in the advertiser's currency.
- **advertiser_currency**: Currency used by the advertiser.
- **channel_id**: Identifier for the ad channel used.
- **channel_name**: Name of the ad channel (e.g., Mobile, Social, Video).
- **max_bid_cpm**: Maximum bid for cost per thousand impressions.
- **network_margin**: Margin of the ad network.
- **campaign_budget_usd**: Budget for the campaign in USD.
- **impressions**: Number of times the ad was displayed.
- **clicks**: Number of clicks the ad received.
- **stats_currency**: Currency used for the ad performance stats.
- **currency_code**: Code for the currency.
- **exchange_rate**: Exchange rate used for currency conversion.
- **media_cost_usd**: Media cost of the ad in USD.
- **search_tag_cat**: Category of the search tag.
- **cmi_currency_code**: Currency code used in cost per impression.
- **timezone**: Timezone of the ad campaign.
- **weekday_cat**: Category indicating if the campaign was run on a weekday or weekend.
- **keywords**: Keywords associated with the ad campaign.

This dataset provides a comprehensive view of ad campaign parameters and their corresponding performance metrics, which will be instrumental in developing models for predicting creative effectiveness and conducting sentiment analysis. The inclusion of metadata such as search tags, landing pages, advertiser details, and keywords further enriches the dataset, enabling more nuanced analysis and feature extraction. *There will be incoherences in the dataset, as the creatives and performance data were pulled from different sources and at different times, this will cause a mismatch in the content of the data inside of the csv against the actual creative media (image or video). Don't worry about this.*

## 4. Methodology

### 4.1. Feature Extraction & Representation Learning

**Visual Representation**
- Extract and analyze visual features from images and videos using computer vision techniques.
    - For images, examples would be:
        - color
        - texture
        - shape
        - composition
        - object detection (i.e. number of objects, predominant objects, etc.)
    - For videos, examples would be:
        - keyframes
        - motion
        - scene transitions
        - object tracking
        - audio analysis
    - Extract high-level embeddings from pre-trained models such as CLIP, ViT.
- Extract high-level embeddings from pre-trained models (CLIP, ViT, EfficientNet, or I3D for video).

**Textual Representation**
- Encode captions, tags via transformer-based embeddings (BERT) or similar.

**Metadata Representation**
- Engineer structured features: platform, budget, targeting parameters.

### 4.2. Modeling Paradigms
- Build a naive baseline, on engineered features.
- More sophisticated approaches, combining visual, textual and metadata feature over multi-modal model configurations using state of the art models.

### 4.3. Visual Sentiment Analysis
   - Analyze visual content and detect emotions using deep learning models.
   - Integrate sentiment analysis for text to provide a comprehensive sentiment score.
   - Include sentiment analysis at a creative level (i.e., inside the image/video).
   - Correlate sentiment intensity and polarity with engagement outcomes to uncover emotional
patterns linked to performance.

### 4.4 Experiment tracking
- Use systems such as Weights & Biases or MLFlow to keep track of the different experiment configurations:
    - Model and hyperparameters.
    - Dataset and feature configuration.
    - Metrics and calibration plots.

### 4.5 Inference service
- Provide a FastApi app with Docker compose. Expose the following:
    - /predict -> accepts a JSON with a `creative_id` and tabular fields required to perform a prediction.
        - Underneath, `creative_id` identifies a sample image from the provided dataset.
        - Should return at least the predicted criteria, sentiment (any feature distribution used).

### 4.6 Architectural decisions and scalability
- Provide an analysis over the different architectural decisions and its tradeoffs.
- Provide an analysis over the current scalability capabilites.
- Provide an analysis over scalability improvements to be taken into consideration to ensure a production-grade environment.

## 5. Evaluation
- **Reproducibility**: 
    - All notebooks can be run top to bottom on a small sample.
    - The service can be executed from the docker compose configuration.
- **Experimentation**: At least 3 sampled runs showing iterations. (baselines -> multimodal -> explanations).
- **Evaluation protocol**:
    - Dataset split discussion, leakage discussion.
    - Justification for metric selection.
    - Ablations and feature analysis.
- **Scalability**: Provide an analysis over current scalability concerns and limitations, proposed improvements and trade-offs between architectural decisions.
- **Reference material**: Provide referenced material to support decisions taken along the execution of the task.

## 6. Deliverables:
- **Image Analysis Pipeline**:
  - A comprehensive Jupyter notebook or Python script that includes all the steps for data preprocessing, feature extraction, and initial data analysis.
  - The script will demonstrate how to extract visual features from images and videos, such as color, texture, shape, composition, keyframes, motion, scene transitions, and object detection.
  - It will also show how to integrate metadata (e.g., captions, tags, performance metrics) into the feature set.
  - Detailed comments and explanations will be provided within the code to ensure clarity and ease of understanding.

- **Forecasting Model Development (Model Selection & Evaluation) Code with Detailed Documentation**:
  - A well-documented codebase for the development of multi-modal models that combine visual and textual data for predicting the effectiveness of ad creatives.
  - The code will include the implementation of visual sentiment analysis models to detect emotions from visual content and text.
  - Scripts for training models to predict key performance metrics, such as the number of clicks, based on the extracted features.
  - An evaluation framework to measure the accuracy and performance of the developed models.
  - Feature importance analysis to understand the impact of each extracted feature on the predictions.
  - Detailed documentation will accompany the code, explaining the model selection process, hyperparameter tuning, and evaluation metrics used.
  - Examples and guidelines for replicating the experiments and extending the models with additional features and functionalities.

- **FastAPI Inference Service(REST):**
    - A production ready FastAPI app exposing:
        - `POST /predict` endpoint that accepts a `creative_id` or `creative_url` and tabular fields required for inference. 
    - Clear input and output models using pydantic schemas 

- **Summary of Insights and Recommendations**:
  - A summary report that highlights the actionable insights derived from the analysis and model predictions.
  - Recommendations for optimizing ad campaigns based on the model predictions and sentiment analysis results.
  - Visualizations and explanations of the key findings, including feature importance, sentiment analysis results, and performance predictions.
  - A detailed explanation of the methodology used, the challenges faced, and the potential improvements for future iterations.


These deliverables will provide a solid foundation for understanding the workflow, from data extraction and feature engineering to model training and evaluation, enabling users to effectively predict ad creative performance, analyze sentiment, and identify key features driving the performance.
