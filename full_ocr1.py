from sanic import Sanic, response
from full_function import full_ocr

app = Sanic(__name__)


@app.route('/full', methods=['POST'])
def full_sanic(request):
    return full_ocr(request)


@app.route('/check')
def health_check(request):
    return response.text('ok')


if __name__ == '__main__':
    app.config.KEEP_ALIVE = False
    app.config.RESPONSE_TIMEOUT = 7200
    app.run(host='0.0.0.0', port=8101)
