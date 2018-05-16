import os
import glob
import numpy as np
from PIL import Image
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io.wavfile
import scipy.signal as signal
from scipy.io import wavfile


composer_list = ['vivaldi', 'bach', 'brahms', 'handel', 'haydn', 'mozart', 'schubert', 'beethoven']
data_types = ['holdout', 'split', 'train']

def wav2spec(wav_file):
	sample_rate, X = scipy.io.wavfile.read(wav_file)
	fig = plt.figure(frameon=False, figsize=(2.56,2.56))
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	spectrum, freq, t, im = plt.specgram(X[:,0], Fs=sample_rate, xextent=(0,20),cmap=cm.get_cmap('viridis'), NFFT=2048)


	plt.axis(ymin=0, ymax=5000)
	#rectangle = plt.Rectangle((6, 0), 8, 5000, fc='#000080')
	#plt.gca().add_patch(rectangle)
	#fig.savefig("test3.png")
	return fig

if __name__ == "__main__":
	for composer in composer_list:
		for data_type in data_types:
			outdir = './reconstructed_spec_stream/' + composer + '/' + data_type
			if not os.path.exists(outdir):
				os.makedirs('./reconstructed_spec_stream/' + composer + '/' + data_type)
				print("made " + composer + ' directory!')
			source_dir = './reconstructed_test_train_midi/' + composer + '/' + data_type + '/'
			listing = sorted(glob.glob(source_dir + '*.mid'))
			for midi_file in listing:
			    fbase = os.path.splitext(os.path.basename(midi_file))[0]
			    print(fbase)
			    inname = source_dir + fbase + ".mid"
			    outname = outdir + '/' + fbase + ".png"
			    # leave one around for each composer at the end
			    tmp = './' + composer + '.wav'
			    logname = outdir + '/' + composer + '.log'
			    # create wav in tmp file
			    os.system('timidity ' + inname + ' -Ow -o ' + tmp + ' >> ' + logname)
			    spec = wav2spec(tmp)
			    spec.savefig(outname)
		print("finished: " + composer + "!")	    

