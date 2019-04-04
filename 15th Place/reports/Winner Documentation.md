# Model documentation and write-up

1. Who are you (mini-bio) and what do you do professionally?

   I am a PhD candidate in Economics at Universitat Pompeu Fabra in Barcelona.

2. High level summary of your approach: what did you do and why?

   The solution consists of four GBM models for each of the four phases. The approach was the following:
    - Generate train sets for each of the phases using the original train set.
    - Generate statistical features (such as min, median, max, standard deviation) for each time series and features derived from the metadata
    - Estimate a [lightGBM](https://github.com/Microsoft/LightGBM) model with five-fold cross-validation
    - Bag the predictions using different seeds to create the folds

3. Copy and paste the 3 most impactful parts of your code and explain what each does and how it helped your model.

    1. You could get better results by estimating the model for each stage separately. This part of the code generates train sets for each of the phases from the original data:
        ```
        phase_list = ['pre_rinse', 'caustic', 'intermediate_rinse', 'acid']
        new_train = []
        for i,p in enumerate(tqdm(phase_list)):
            df = train[(train['phase'].isin(phase_list[:i+1]))].copy()
            df['max_phase'] = p
            df['last_phase'] = df['process_id'].map(df.groupby('process_id')['phase'].last())
            df['orig_process_id'] = df['process_id'].copy()
            df['process_id'] = df['max_phase']+'-'+df['process_id'].astype('str')
            new_train.append(df)
        train = pd.concat(new_train)
        ```

    2. Using `dart` mode instead of the default `gbdt` in lightGBM gave an over 25% improvement in score:
        ```
        params = {
            'boosting_type': 'dart',
        ...
        ```
        [DART](https://arxiv.org/abs/1505.01866) is slower than the default method, however given the data size this was not an issue.

    3. Using a custom metric to fit the model also gave a boost to the score:
    ```
    def rinse_mape(y, preds):
        return np.mean(abs(y-preds)/np.maximum(y,290000))

    def rinse_mape_lgb(preds, train_data):
        y = train_data.get_label()
        return 'loss', rinse_mape(y, preds), False
    ```

4. What are some other things you tried that didn’t necessarily make it into the final workflow (quick overview)?

    Unfortunately I was not able to spend a lot of time on this competition, however, some of the things
    I have tried that did not work included neural nets, simple time series models, deriving various features from the time series and
    predicting the turbidity of the first part of the final stage and using it as a feature in the main model.

5. Did you use any tools for data preparation or exploratory data analysis that aren’t listed in your code submission?

    To improve the performance of the model I used permutation importance for feature selection (using [eli5 package](https://github.com/TeamHG-Memex/eli5))
    and [Bayesian optimisation](https://github.com/fmfn/BayesianOptimization) for hyperparameter selection.

6. How did you evaluate performance of the model other than the provided metric, if at all?

    To track the progress I used the five-fold cross-validation loss for each phase weighted 10%/30%/30%/30% as in the competition metric.

7. Anything we should watch out for or be aware of in using your model (e.g. code quirks, memory requirements, numerical stability issues, etc.)?

    No.

8. Do you have any useful charts, graphs, or visualizations from the process?

    No.

9. If you were to continue working on this problem for the next year, what methods or techniques might you try in order to build on your work so far? Are there other fields or features you felt would have been very helpful to have?

    `object_id` was the most important feature in the model, so looking deeper into characteristics of each object could be promising, especially if the goal is to eventually make predictions on the objects with very few observations.