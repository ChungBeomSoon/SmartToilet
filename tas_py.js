/**
 * Created by ryeubi on 2015-08-31.
 * Updated 2017.03.06
 * Made compatible with Thyme v1.7.2
 */




/* USER CODE */
conf = require('./conf.js');
let moment = require('moment');

var Onem2mClient = require('./onem2m_client');
// var thyme_tas = require('./thyme_tas');

var options = {
    protocol: conf.useprotocol,
    host: conf.cse.host,
    port: conf.cse.port,
    mqttport: conf.cse.mqttport,
    wsport: conf.cse.wsport,
    cseid: conf.cse.id,
    aei: conf.ae.id,
    aeport: conf.ae.port,
    bodytype: conf.ae.bodytype,
    usesecure: conf.usesecure,
};

onem2m_client = new Onem2mClient(options);

let sendDataTopic = {
    stool: '/thyme/stool',
    constipation: '/thyme/constipation',
    habbit: '/thyme/habbit',
};


let customCreateCin = (topic, message) => {
    let content = null;
    let parent = null;

    if(topic == sendDataTopic.stool) {
        parent = conf.cnt[0].parent + '/' + conf.cnt[0].name;
        let curTime =  moment().format();
        let curVal = parseFloat(message.toString()).toFixed(1);
        content = {
            t: curTime,
            v: curVal
        };
    }
    else if(topic === sendDataTopic.constipation) {
        parent = conf.cnt[1].parent + '/' + conf.cnt[1].name;
        let curTime =  moment().format();
        let curVal = parseFloat(message.toString()).toFixed(1);
        content = {
            t: curTime,
            v: curVal
        };
    }
    else if(topic === sendDataTopic.habbit) {
        parent = conf.cnt[2].parent + '/' + conf.cnt[2].name;
        let curTime =  moment().format();
        let curVal = parseFloat(message.toString()).toFixed(1);
        content = {
            t: curTime,
            v: curVal
        };
    }

    if(content !== null) {
        onem2m_client.create_cin(parent, 1, JSON.stringify(content), this, function (status, res_body, to, socket) {
            console.log('x-m2m-rsc : ' + status + ' <----');
        });
    }
}

setInterval(()=>{
    console.log('start python');
    let command = 'python evaluatePOOP.py';
    const result = require('child_process').execSync(command);
    let ret = result.toString().replace('\n', '').split(',');
    let stool = ret[0];
    let constipation = ret[1];
    let habbit = ret[2];
    console.log('stool : ', stool, 'constipation : ', constipation, 'habbit : ', habbit);
    customCreateCin(sendDataTopic['stool'], stool);
    customCreateCin(sendDataTopic['constipation'], constipation);
    customCreateCin(sendDataTopic['habbit'], habbit);
}, 3000);

/* */
