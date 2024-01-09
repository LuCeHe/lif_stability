import numpy as np


def save_sentences(task, text_path, gen):
    if gen.name in ['ptb', 'time_ae', 'time_ae_merge', 'wmt14', 'wordptb']:
        o_n = task['target_output'].any(axis=-1).astype(int)
        i_n = task['input_spikes'].any(axis=-1).astype(int)
        if 'time_ae' in gen.name:
            output_s = o_n * (np.argmax(task['output_net'], axis=-1) + 1)
            target_s = o_n * (np.argmax(task['target_output'], axis=-1) + 1)
            input_s = i_n * (np.argmax(task['input_spikes'], axis=-1) + 1)
            union_string = ' '
        else:
            output_s = np.argmax(task['output_net'], axis=-1)
            target_s = task['target_output']
            input_s = np.squeeze(task['input_spikes'], axis=-1)
            union_string = '' if not gen.name == 'wordptb' else ' '
        input_s = input_s[:, ::gen.timerepeat]
        np.save(text_path, output_s)

        with open(text_path + '_sentences.txt', 'w', encoding="utf-8") as f_s, \
                open(text_path + '_indices.txt', 'w', encoding="utf-8") as f_i:
            for i in range(output_s.shape[0]):
                o_i = output_s[i]
                t_i = target_s[i]
                i_i = input_s[i]

                try:
                    o_s = gen.decode([o_i])[0].replace('[PAD]', '')
                    t_s = gen.decode([t_i])[0].replace('[PAD]', '')
                    i_s = gen.decode([i_i])[0].replace('[PAD]', '')
                except:
                    o_s = union_string.join([gen.id_to_word[i] for i in o_i]).replace('<PAD> ', '')
                    t_s = union_string.join([gen.id_to_word[i] for i in t_i]).replace('<PAD> ', '')
                    i_s = union_string.join([gen.id_to_word[i] for i in i_i]).replace('<PAD> ', '').replace('<START> ', '')

                f_s.write('\n\noutput_sentence: \n--- {}'.format(o_s))
                f_s.write('\ntarget_sentence: \n--- {}'.format(t_s))
                f_s.write('\ninput_sentence: \n--- {}'.format(i_s))
                f_i.write('\n\noutput_sentence: \n--- {}'.format(o_i))
                f_i.write('\ntarget_sentence: \n--- {}'.format(t_i))
                f_i.write('\ninput_sentence: \n--- {}'.format(i_i))

