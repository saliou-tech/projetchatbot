from flask import Flask ,jsonify,request, send_from_directory,render_template, Response, send_file, make_response
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
from ChatBot import Chatbot

@app.route('/chatbot', methods=['POST'])
def get_chatbot_answer():
    # bar=dataframe.covid('Covid19SN_datas.xlsx')
    chatbot=Chatbot()

    values = request.json
    print(values['values'])
    chatbot_response=chatbot.getQuestionGiveAnswer(values['values'],chatbot.vocab,chatbot.enc_model,chatbot.dec_model,chatbot.dense,chatbot.inv_vocab)
    #print(chatbot_response)
   
    response = {
        'data': chatbot_response
    }
    return jsonify(response) , 200



# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=port, debug=True)
    app.run(host = '0.0.0.0' ,port = 5000 )
