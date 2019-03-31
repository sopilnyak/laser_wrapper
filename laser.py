import os
os.environ['LASER'] = 'LASER'

from LASER.source.embed import SentenceEncoder, EncodeFile, EmbedLoad
from LASER.source.lib.text_processing import Token, BPEfastApply
import tempfile


class Laser:

    def __init__(self, encoder_path, bpe_codes, use_gpu=True):
        """
        :param encoder_path: path to encoder weights
        :param bpe_codes: path to file with BPE codes
        :param use_gpu: use GPU instead of CPU
        """
        self.encoder = SentenceEncoder(encoder_path,
                                       max_sentences=None,
                                       max_tokens=12000,
                                       sort_kind='quicksort',
                                       cpu=not use_gpu)
        self.bpe_codes = bpe_codes

    def __call__(self, sentences, tokenizer_lang='en', verbose=False):
        """
        Returns LASER embeddings of input sentences.
        :param sentences: list of strings with input sentences
        :param tokenizer_lang: language to perform tokenization with
        :param verbose: show log messages
        :return: (n_sentences, 1024)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Writing sentences to file
            input_file = os.path.join(tmpdir, 'input')
            with open(input_file, 'w') as f:
                for sentence in sentences:
                    f.write("%s\n" % sentence.replace('\n', ' '))

            # Tokenizing
            token_file = os.path.join(tmpdir, 'token')
            Token(input_file,
                  token_file,
                  lang=tokenizer_lang,
                  romanize=False,
                  lower_case=True, gzip=False,
                  verbose=verbose, over_write=False)

            # Applying bpe
            bpe_file = os.path.join(tmpdir, 'bpe')
            BPEfastApply(token_file,
                         bpe_file,
                         self.bpe_codes,
                         verbose=verbose, over_write=False)

            # Getting embeddings
            emb_file = os.path.join(tmpdir, 'emb')
            EncodeFile(self.encoder,
                       bpe_file,
                       emb_file,
                       verbose=verbose, over_write=False,
                       buffer_size=10000)

            return EmbedLoad(emb_file, verbose=verbose)
