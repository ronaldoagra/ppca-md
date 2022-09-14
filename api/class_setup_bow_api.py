
    import pandas as pd
    from pycaret.MLUsecase.CLASSIFICATION import load_model, predict_model
    from fastapi import FastAPI
    import uvicorn
    # Create the app
    app = FastAPI()
    # Load trained Pipeline
    model = load_model('api/class_setup_bow_api')
    # Define predict function
    @app.post('/predict')
    def predict(input):
        data = pd.DataFrame([[input]])
        data.columns = ['input']
        predictions = predict_model(model, data=data)
        return {'prediction': list(predictions['Label'])}
    if __name__ == '__main__':
        uvicorn.run(app, host='127.0.0.1', port=8000)