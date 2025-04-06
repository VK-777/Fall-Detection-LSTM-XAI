from Constants import constants
from Services.XAI_MobiFall_LSTM import train_model_mobifall, display_result_mobifall
from Services.XAI_K_fall import train_model_k, explain_model_k

if __name__ == "__main__":
    train_model_mobifall()
    display_result_mobifall()
    train_model_k()
    explain_model_k()



