import os
import glob

composer_list = ['vivaldi', 'bach', 'brahms', 'handel', 'haydn', 'mozart', 'schubert', 'beethoven']

for composer in composer_list:
	if not os.path.exists('./reconstructed_wav/' + composer):
		os.makedirs('./reconstructed_wav/' + composer)
		print("made " + composer + ' directory!')
	source_dir = './reconstructed_midi/' + composer + '/'
	listing = sorted(glob.glob(source_dir + '*.mid'))
	for midi_file in listing:
	    fbase = os.path.splitext(os.path.basename(midi_file))[0]
	    print(fbase)
	    inname = source_dir + fbase + ".mid"
	    outname = './reconstructed_wav/' + composer + '/' + fbase + ".wav"
	    logname = './reconstructed_wav/' + composer + '.log'
	    os.system('timidity ' + inname + ' -Ow -o ' + outname + ' >> ' + logname)
	    for line in open(logname, 'r'):
	    	if 'Notes lost totally:' in line:
	    		num_notes = int(line.split(':')[1].strip())
	    		if num_notes > 0:
	    			f = open('./reconstructed_wav/errors.log', 'a+')
	    			f.write(outname)