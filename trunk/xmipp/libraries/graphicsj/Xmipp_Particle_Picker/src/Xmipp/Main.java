package Xmipp;

import java.util.logging.Level;
import javax.swing.SwingUtilities;

import model.ExecutionEnvironment;

import gui.ParticlePickerJFrame;
import ij.ImagePlus;





class Main implements Runnable
{
	//0 --> input metadata
	//1 --> output dir
	//2 --> number of threads for automatic picking
	//3 --> fast mode for automatic picking
	//4 --> incore for automatic picking
	public static void main(String[] args)
	{
		String mgselfile = args[0];
		String outputdir = args[1];
		
		if(args.length == 2)
			SwingUtilities.invokeLater(new Main(mgselfile, outputdir));
		else
		{
			int threads = Integer.parseInt(args[2]);
			boolean fastMode = Boolean.parseBoolean(args[3]);
			boolean incore = Boolean.parseBoolean(args[4]);
			SwingUtilities.invokeLater(new Main(mgselfile, outputdir, threads, fastMode, incore));
		}
			
	}
	
	
	Main(String mgselfile, String outputdir)
	{
		ExecutionEnvironment.setMicrographsSelFile(mgselfile);
		ExecutionEnvironment.setOutputDir(outputdir);
		ExecutionEnvironment.setIsAuto(false);
	}
	
	Main(String mgselfile, String outputdir, int threads, boolean fastMode, boolean incore)
	{
		
		ExecutionEnvironment.setMicrographsSelFile(mgselfile);
		ExecutionEnvironment.setOutputDir(outputdir);
		ExecutionEnvironment.setThreads(threads);
		ExecutionEnvironment.setFastMode(fastMode);
		ExecutionEnvironment.setIncore(incore);
		ExecutionEnvironment.setIsAuto(true);
	}
	
	@Override
	public void run() {
		try {
			
			ParticlePickerJFrame frame = new ParticlePickerJFrame();
			
		} catch (Exception e) {
			ExecutionEnvironment.getLogger().log(Level.SEVERE, e.getMessage(), e);
		}
	}
}