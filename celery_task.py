from celery import Celery
import requests
import json
import os
import time
import socket
import urllib
from kombu import Exchange,Queue
ddwork = Celery('sh_tasks', broker='redis://172.30.81.208:30443/0', backend='redis://172.30.81.208:30443/1')
ddwork.conf.CELERY_QUEUES = (
Queue("default",Exchange("default"),routing_key="default"),
Queue("for_nlp_api_callback",Exchange("for_nlp_api_callback"),routing_key="for_nlp_api_callback"),
Queue("for_fce_callback",Exchange("for_fce_callback"),routing_key="for_fce_callback"),
Queue("for_ocr_callback",Exchange("for_ocr_callback"),routing_key="for_ocr_callback"),
Queue("for_fre_type_all_callback",Exchange("for_fre_type_all_callback"),routing_key="for_fre_type_all_callback"),
Queue("for_two_class_callback",Exchange("for_two_class_callback"),routing_key="for_two_class_callback"),
Queue("for_nlp_get_callback",Exchange("for_nlp_get_callback"),routing_key="for_nlp_get_callback"),
Queue("for_nlp_type_callback",Exchange("for_nlp_type_callback"),routing_key="for_nlp_type_callback")
)

ddwork.conf.CELERY_ROUTES = {
'celery_task.nlp_api_callback':{"queue":"for_nlp_api_callback","routing_key":"for_nlp_api_callback"},
'celery_task.fce_callback':{"queue":"for_fce_callback","routing_key":"for_fce_callback"},
'celery_task.ocr_callback':{"queue":"for_ocr_callback","routing_key":"for_ocr_callback"},
'celery_task.fre_type_all_callback':{"queue":"for_fre_type_all_callback","routing_key":"for_fre_type_all_callback"},
'celery_task.two_class_callback':{"queue":"for_two_class_callback","routing_key":"for_two_class_callback"},
'celery_task.nlp_get_callback':{"queue":"for_nlp_get_callback","routing_key":"for_nlp_get_callback"},
'celery_task.nlp_type_callback':{"queue":"for_nlp_type_callback","routing_key":"for_nlp_type_callback"},
}

ddwork.conf.CELERYD_FORCE_EXECV = True

ddwork.conf.BROKER_TRANSPORT_OPTIONS = {'visibility_timeout': 36000}

requests.adapters.DEFAULT_RETRIES = 3 #重连次数

@ddwork.task
def fce_callback(url, odoo_url, data, task_id, work_type, token, dbname, state, project_id, timeout=36000):
    print ('URL:{},参数:{}'.format(url,data),task_id)
    odoo_data = {'access_token':token,'dbname':dbname}
    odoo_url = 'http://172.30.81.208:32621/restapi/1.0/extension_object/sh.work.task/ML_action_to_fce_callback'
    headers = {'User-Agent': 'User-Agent:Mozilla/5.0'}
    start_time = time.time()
    try:
        res = requests.post("http://127.0.0.1:8005", data=data, headers=headers, timeout=timeout).text
        print ('URL:{},返回值:{},类型:{}'.format(url,res,type(res)),task_id)
        #回调
        if work_type == 'fce':    
            odoo_data.update({'result':res,'task_id':task_id,'all_time':str(time.time()-start_time),'state':state})
        odoo_res = requests.post(odoo_url, data=odoo_data, headers=headers, timeout=timeout).text
        print('回调返回值：{}'.format(json.loads(odoo_res)))
        return odoo_data
    except Exception as e:
        if not os.path.exists(r'/home/ddwork/projects/compound_log'):
            os.makedirs('/home/ddwork/projects/compound_log')
        if not os.path.exists(r'/home/ddwork/projects/compound_log/{}'.format(project_id)):
            os.makedirs('/home/ddwork/projects/compound_log/{}'.format(project_id))
        if not os.path.exists(r'/home/ddwork/projects/compound_log/{}/FCE'.format(project_id)):
            os.makedirs('/home/ddwork/projects/compound_log/{}/FCE'.format(project_id))
        with open(r'/home/ddwork/projects/compound_log/{}/FCE/{}_log.txt'.format(project_id,str(time.strftime('%Y-%m-%d'))), 'a', encoding='utf-8') as gap:
            gap.write(str(time.strftime('%Y-%m-%d %H:%M:%S'))+'\n'+ str(url) +'\n'+ str(data) + '\n'+str(task_id) +'\n'+str(e)+'\n\n')
            if odoo_url:
                gap.write(str(time.strftime('%Y-%m-%d %H:%M:%S'))+'\n'+ str(odoo_url) +'\n'+ str(odoo_data) + '\n'+str(task_id) +'\n'+str(e)+'\n\n')
            gap.close() 
        odoo_data.update({'result':{"res":str(e)},'task_id':task_id,'all_time':str(time.time()-start_time),'state':state})
        odoo_res = requests.post(odoo_url, data=odoo_data, headers=headers, timeout=timeout).text
        print('回调异常返回值：{}'.format(json.loads(odoo_res)))    
        return {'result':str(e),'task_id':task_id,'start_time':start_time}
