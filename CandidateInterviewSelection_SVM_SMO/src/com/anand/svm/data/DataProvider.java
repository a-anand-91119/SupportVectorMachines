package com.anand.svm.data;

import java.util.concurrent.ThreadLocalRandom;

public class DataProvider {

	private static final double[][][] TRAINING_DATA = new double[][][] {
		{{6.881596019909946, 6.433391000070089}, {+1}},
		{{8.803881022694464, 6.338354348096247}, {+1}},
		{{2.1018459756918473, 2.9953087530367806}, {-1}},
		{{2.400323190074236, 0.5209916801883718}, {-1}},
		{{2.2886572043588895, 2.1801499037562175}, {-1}},
		{{0.17800582749854632, 0.9595618504946835}, {-1}},
		{{6.01201070762595, 6.620586139193688}, {+1}},
		{{0.9420285299189135, 1.6259789190130847}, {-1}},
		{{7.756650831097883, 9.06520410648138}, {+1}},
		{{7.220801357414227, 6.724347227395154}, {+1}},
		{{9.737438214419491, 7.491482106161717}, {+1}},
		{{9.241966237635491, 8.16653394752}, {+1}},
		{{8.746752725486274, 7.40830875767516}, {+1}},
		{{0.3220449732313191, 0.649337186865605}, {-1}},
		{{6.862093025468936, 9.01174669521017}, {+1}},
		{{1.142023695598804, 2.8822531108557103}, {-1}},
		{{1.6158607688657456, 2.275424318643211}, {-1}},
		{{7.538257441309809, 7.764741341334887}, {+1}},
		{{0.05799455138813525, 2.429592626225856}, {-1}},
		{{7.09092244594016, 9.481121619379424}, {+1}},
		{{9.115500425462601, 6.840914323126027}, {+1}},
		{{9.433251564427161, 8.818513369382298}, {+1}},
		{{9.096268517466989, 6.382870343074229}, {+1}},
		{{9.110547649935882, 7.603298177099598}, {+1}},
		{{2.923204206785396, 2.2211762197303084}, {-1}},
		{{6.789941306640914, 9.009137305426849}, {+1}},
		{{6.063596720010192, 6.354662447603653}, {+1}},
		{{0.03817309337725838, 2.2841859129289013}, {-1}},
		{{0.5144751332309239, 1.323688198155487}, {-1}},
		
		/*{{2.9334258007919614, 1.6616389476613647}, {-1}},
		{{3.194823649593096, 8.202620098127335}, {+1}},
		{{5.823765087779308, 8.100905131339026}, {+1}},
		{{4.693974905955097, 3.582068722870572}, {-1}},
		{{1.4255456492844731, 6.472889775865869}, {-1}},
		{{7.029029195667272, 5.416190423220292}, {+1}},
		{{1.3700034903965064, 5.127661532447572}, {-1}},
		{{8.729607169498214, 3.877261700342205}, {+1}},
		{{7.509839588271397, 4.4018306117963775}, {+1}},
		{{4.808863892896182, 5.164842313741815}, {+1}},
		{{6.14157807661479, 2.4745694037512624}, {-1}},
		{{7.921391094185882, 3.244619756482141}, {+1}},
		{{4.796729106494203, 0.5343883494580515}, {-1}},
		{{8.565002074887758, 6.91800910523776}, {+1}},
		{{5.618231529804216, 6.496858786035878}, {+1}},
		{{6.703062945256003, 7.9331403547648645}, {+1}},
		{{6.266450827781858, 3.9078676028266877}, {+1}},
		{{8.446989678680929, 6.163478434523554}, {+1}},
		{{5.74959838573453, 2.8738009615359736}, {-1}},
		{{8.77076360238316, 6.277155792396766}, {+1}},
		{{9.908020194390136, 5.184293224924509}, {+1}},
		{{8.165452393890929, 9.648034450822934}, {+1}},
		{{8.827385184930897, 6.652103519169619}, {+1}},
		{{0.48733672120426896, 7.853254912782212}, {-1}},
		{{9.992994354621775, 3.4507062691372914}, {+1}},
		{{0.43417334219995385, 0.3269695129036665}, {-1}},
		{{5.359072901117702, 0.3186036603669684}, {-1}},
		{{5.774117952011038, 0.47635085238661445}, {-1}},
		{{0.11922429912650134, 7.2983687650502915}, {-1}},
		{{3.2453565265938042, 6.943643975406671}, {+1}},
		{{6.895610878059568, 7.345190583501951}, {+1}},
		{{4.356027456151301, 2.0421824202174923}, {-1}},
		{{0.3855332756347396, 4.672771747863246}, {-1}},
		{{0.6433666131903848, 5.262912588620512}, {-1}},
		{{8.739777237492106, 4.515581654336735}, {+1}},
		{{2.7061063541722365, 5.600208971493602}, {-1}},
		{{1.7437115306438111, 3.564308055343309}, {-1}},
		{{4.088535929679268, 5.636936108694263}, {+1}},
		{{8.874411921037854, 8.61036045271409}, {+1}},
		{{1.7009272372097828, 6.6324967472970355}, {-1}}*/

	};

	public static double[][][] getTrainingData() {
		return TRAINING_DATA;
	}

	public static int getTrainingDataSize() {
		return TRAINING_DATA.length;
	}
	
	public static void main(String[] args) {
		double firstMark = 0.0;
		double secondMark = 0.0;
		for(int i =0;i<40;i++) {
			firstMark = ThreadLocalRandom.current().nextDouble(0.0, 10.0);
			secondMark = ThreadLocalRandom.current().nextDouble(0.0, 10.0);
			//if((firstMark > 6 && secondMark > 6) || (firstMark < 3 && secondMark < 3))
				System.out.println("{{"+firstMark+", " + secondMark + "}, {"+getSelectedOrNot(firstMark, secondMark)+"}},");
		}
		
	}

	private static String getSelectedOrNot(double firstMark, double secondMark) {
		if(firstMark < 3.0 || secondMark < 3.0 || (firstMark + secondMark ) / 2.0 < 4.5)
			return "-1";
		
		return "+1";
	}
	
}