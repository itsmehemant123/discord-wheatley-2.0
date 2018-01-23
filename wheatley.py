import argparse
import codecs
import time
import logging
import re
from random import random

import tensorflow as tf

from nmt.nmt import *
import nmt.attention_model as attention_model
import nmt.gnmt_model as gnmt_model
import nmt.model as nmt_model
import nmt.model_helper as model_helper
from nmt.utils import misc_utils as utils
from nmt.utils import nmt_utils

class Wheatley:

    def __init__(self, bot):
        logging.basicConfig(level=logging.INFO)
        out_dir = '<path_to_model>'

        self.ping_replace = re.compile(r"<@![0-9]{2,}>", re.IGNORECASE)
        self.bot = bot

        nmt_parser = argparse.ArgumentParser()
        add_arguments(nmt_parser)
        flags, unparsed = nmt_parser.parse_known_args()
        default_hparams = create_hparams(flags)
        self.hparams = create_or_load_hparams(
            out_dir, default_hparams, flags.hparams_path, save_hparams=False)
        ckpt = tf.train.latest_checkpoint(out_dir)

        if not self.hparams.attention:
            model_creator = nmt_model.Model
        elif self.hparams.attention_architecture == "standard":
            model_creator = attention_model.AttentionModel
        elif self.hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
            model_creator = gnmt_model.GNMTModel
        else:
            raise ValueError("Unknown model architecture")
        self.infer_model = model_helper.create_infer_model(
            model_creator, self.hparams, None)
        self.session = tf.InteractiveSession(graph=self.infer_model.graph, config=utils.get_config_proto())
        self.loaded_infer_model = model_helper.load_model(self.infer_model.model, ckpt, self.session, "infer")
        
        # with tf.Session(graph=self.infer_model.graph, config=utils.get_config_proto()) as sess:
        #     self.loaded_infer_model = model_helper.load_model(
        #         self.infer_model.model, ckpt, sess, "infer")

    def shutdown(self):
        logging.info('Shutting down wheatley.')
        self.session.close()

    async def talk(self, message):
        logging.info('MSG: ' + message.content + ' in ' + message.channel.name)

        luck = random.random()
        if (luck > 0):
            logging.info('trigerred')
            start_time = time.time()
            try:
                await self.bot.send_typing(message.channel)
            except:
                pass  # fuck it
            
            
            self.session.run(self.infer_model.iterator.initializer,
                    feed_dict={
                        self.infer_model.src_placeholder: [message.content],
                        self.infer_model.batch_size_placeholder: self.hparams.infer_batch_size
                    })
            num_translations_per_input = max(
                min(self.hparams.num_translations_per_input, self.hparams.beam_width), 1)

            nmt_outputs, _ = self.loaded_infer_model.decode(self.session)
            if self.hparams.beam_width == 0:
                nmt_outputs = np.expand_dims(nmt_outputs, 0)

            batch_size = nmt_outputs.shape[1]

            for sent_id in range(batch_size):
                for beam_id in range(num_translations_per_input):
                    response = nmt_utils.get_translation(
                        nmt_outputs[beam_id],
                        sent_id,
                        tgt_eos=self.hparams.eos,
                        subword_option=self.hparams.subword_option)
                    end_time = time.time()
                    logging.info('Time taken for response:' +
                                str(end_time - start_time))

                    #clean_msg = self.ping_replace.sub('', response)
                    clean_msg = str(
                        response, 'utf-8').replace('<unk>', '').replace('\n', '').strip()
                    await self.bot.send_message(message.channel, clean_msg)

            
