from music21 import stream, note, chord, midi, instrument
from music21.pitch import PitchException, AccidentalException
from music21.instrument import *
import os


def text_to_mid(source_file, dest_dir, dur=True, poly=True, notelength='eighth', output_instrument=Piano()):
    with open(source_file) as f:
        lines = f.readlines()
    
    # Check if destination exists and create if not
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    total = len(lines)
    i = 1
    for line in lines:
        s1 = stream.Stream()
        s1.voices
        if not poly and not dur:
            for token in line.strip().split(' '):
                try:
                    s1.append(note.Note(token, type=notelength))
                except (PitchException, AccidentalException):
                    continue
        elif poly and dur:
            for chord_token in line.strip().split(' '):
                chord_token = chord_token.split('.')
                notelength = chord_token.pop()
                try:
                    s1.append(chord.Chord(chord_token, type=notelength))
                except (PitchException, AccidentalException):
                    continue
        else:
            # can deal with later if necessary
            print("case not handled")
        s1.insert(0, output_instrument)
        s1.write('midi', os.path.join(dest_dir, 
                                      os.path.splitext(os.path.basename(source_file))[0] + 
                                      '_' + str(i) + '.mid')
                )
        print("Created midi file", i, "out of", total)
        i += 1

if __name__ == '__main__':
    for composer in ['vivaldi', 'bach', 'brahms', 'handel', 'haydn', 'mozart', 'schubert', 'beethoven']:
        for data_type in ['holdout', 'split', 'train']:
            dest = './reconstructed_test_train_midi/' + composer + '/' + data_type
            source = './train-test/' + composer + '_mono_' + data_type + '.txt'
            text_to_mid(source, dest, dur=False, poly=False)
            print('Completed' + composer + '!')


