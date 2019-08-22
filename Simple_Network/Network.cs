using System;
using System.IO;
using System.Linq;
using DFMatrix;
using SimpleNetworks;

namespace Simple_NetworkANN {
    static class Program {
        static void Main(string[] args) {
            SimpleANN nn = null;
            string path = Directory.GetCurrentDirectory() + "/Networks/";
            string ImageFileName = "t10k-images.idx3-ubyte";
            string LabelFileName = "t10k-labels.idx1-ubyte";
           // string ImageFileName = "train-images.idx3-ubyte";
            //string LabelFileName = "train-labels.idx1-ubyte";


            // First we need to get the data
            FileStream Training_Data = new FileStream("MnistDatasets/"+ImageFileName, FileMode.Open);
            FileStream Training_Labels = new FileStream("MnistDatasets/"+LabelFileName, FileMode.Open);
            BinaryReader data_reader = new BinaryReader(Training_Data);
            BinaryReader labals_reader = new BinaryReader(Training_Labels);
            int Magic_Number = data_reader.ReadBigInt32();
            int Input_Num = data_reader.ReadBigInt32();
            int Pixel_Rows = data_reader.ReadBigInt32();
            int Pixel_Cols = data_reader.ReadBigInt32();

            Magic_Number = labals_reader.ReadBigInt32();
            int Labels_Num = labals_reader.ReadBigInt32();

            Vector[] Inputs = new Vector[Input_Num];
            Vector[] Targets = new Vector[Labels_Num];


            Random random = new Random();

            // We assume we have an imputs vector for every label
            for (int i = 0; i < Input_Num; i++) {
                Inputs[i] = new Vector(Pixel_Rows * Pixel_Cols);
                Targets[i] = new Vector(10);

                for (int j = 0; j < Pixel_Rows; j++) {
                    for (int k = 0; k < Pixel_Cols; k++) {
                        Inputs[i][k * Pixel_Cols + j] = data_reader.ReadByte() / (double)255;
                    }
                }


                int label = labals_reader.ReadByte();



                for (int j = 0; j < Targets[i].Length; j++) {
                    if (label == Targets[i].Length) {
                        Targets[i][Targets[i].Length - 1] = 1;
                        break;
                    }

                    if (label == j) {
                        Targets[i][j] = 1;
                    }
                    else {
                        Targets[i][j] = 0;
                    }
                }
            }


            data_reader.Close();
            labals_reader.Close();
            Training_Data.Close();
            Training_Labels.Close();


            string consoleRead = "";
            bool correct = false;
            while (!correct) {
                Console.WriteLine("Load saved network?");
                consoleRead = Console.ReadLine();
                consoleRead = consoleRead.ToLower();
                if (consoleRead == "yes" || consoleRead == "no" || consoleRead == "y" || consoleRead == "n") {
                    correct = true;
                }
                else {
                    Console.WriteLine("Please enter: yes, no, y, or n....");
                }
            }
            correct = false;
            consoleRead = consoleRead.Replace("yes", "y");
            consoleRead = consoleRead.Replace("no", "n");
            switch (consoleRead) {
                case "y":
                    if (Directory.Exists(path)) {
                        string[] files = Directory.GetFiles(path);
                        if (files.Length > 0) {
                            int saveNum = -1;
                            while (saveNum < 0 || saveNum >= files.Length) {
                                Console.WriteLine("Enter the number according to the file: ");
                                for (int i = 0; i < files.Length; i++) {
                                    Console.WriteLine("---------------------------------------------");
                                    Console.WriteLine(i + " | " + Path.GetFileName(files[i]));
                                    Console.WriteLine("---------------------------------------------");
                                }
                                consoleRead = Console.ReadLine();
                                Int32.TryParse(consoleRead, out saveNum);
                            }
                            nn = SimpleANN.FromJson(files[saveNum]);
                            if (nn != null) {
                                break;
                            }
                        }
                    }
                    Console.WriteLine("Cannot create file. Either Directory/file is missing or the file cannot be loaded properly.");
                    goto case "n";
                case "n":
                    Console.WriteLine("Creating new network based on current data loaded.");
                    nn = new SimpleANN(Inputs[0].Length, Targets[0].Length, 1, 16);
                    break;
            }
            if (nn != null) {
                nn.Learning_Rate = .5f;
                nn.Init();
            }

            while (!correct) {
                Console.WriteLine("Are we training?.....\n");
                consoleRead = Console.ReadLine();
                consoleRead = consoleRead.ToLower();
                if (consoleRead == "yes" || consoleRead == "no" || consoleRead == "y" || consoleRead == "n") {
                    correct = true;
                }
                else {
                    Console.WriteLine("Please enter: yes, no, y, or n....");
                }
            }
            correct = false;
            consoleRead = consoleRead.Replace("yes", "y");
            consoleRead = consoleRead.Replace("no", "n");
            switch (consoleRead) {
                case "y":
                    Console.WriteLine("How many training interations?.....\n"); consoleRead = Console.ReadLine();
                    int train_itr = 0;
                    Int32.TryParse(consoleRead, out train_itr);


                    //Console.Out.WriteLine(Inputs);
                    //Console.Out.WriteLine(Targets);

                    //Train
       
                    for (int i = 0; i < train_itr; i++) {

                        drawTextProgressBar(i, train_itr);
                        nn.Train(Inputs, Targets);

                               
                        

                        Console.Out.WriteLine("Training itr: " + i + " || Error: " + nn.Loss * 100 + "%");
                    }
                    drawTextProgressBar(train_itr, train_itr);
                    break;

                case "n":
                    Console.Out.WriteLine("Skipping training.....");
                    break;
            }

            bool stop = false;
            double total_correct = 0;
            int count = 1;
            int rand = 0;
            int max = 0;
            while (!stop) {
                 rand = random.Next(0, Inputs.Length);
                Vector Outputs = nn.Predict(Inputs[rand]);
                max = 0;
                for (int i = 0; i < Outputs.Length; i++) {
                    if (Outputs[i] > Outputs[max]) {
                        max = i;
                    }
                }
                Console.Out.WriteLine("Prediction Vector: " + Outputs);
                int target = 0;
                for (int i = 0; i < Targets[rand].Length; i++) {
                    if (Targets[rand][i] == 1) {
                        target = i;
                    }
                }
          
                Console.Out.WriteLine("Target Vector: " + Targets[rand]);
                //total_error += nn.CalcError(Outputs, Targets[rand]);

                Console.Out.WriteLine("Prediction: " + max);
                Console.Out.WriteLine("Target: " + target);
                if (max == target) {
                    total_correct++;
                }
                Console.Out.WriteLine("Accuracy: " + (total_correct / (float)count) * 100 + "%");
                count++;
                correct = false;
                while (!correct) {
                    Console.WriteLine("Should we guess again?");
                    consoleRead = Console.ReadLine();
                    consoleRead = consoleRead.ToLower();
                    if (consoleRead == "yes" || consoleRead == "no" || consoleRead == "y" || consoleRead == "n") {
                        correct = true;
                    }
                    else {
                        Console.WriteLine("Please enter: yes, no, y, or n....");
                    }
                }
                correct = false;
                consoleRead = consoleRead.Replace("yes", "y");
                consoleRead = consoleRead.Replace("no", "n");
                switch (consoleRead) {
                    case "y":
                        break;
                    case "n":
                        stop = true;
                        break;
                }
            }

            correct = false;
            while (!correct) {
                Console.WriteLine("Should we save the network?");
                consoleRead = Console.ReadLine();
                consoleRead = consoleRead.ToLower();
                if (consoleRead == "yes" || consoleRead == "no" || consoleRead == "y" || consoleRead == "n") {
                    correct = true;
                }
                else {
                    Console.WriteLine("Please enter: yes, no, y, or n....");
                }
            }
            correct = false;
            consoleRead = consoleRead.Replace("yes", "y");
            consoleRead = consoleRead.Replace("no", "n");
            switch (consoleRead) {
                case "y":
                    Console.WriteLine("What should we name it?");
                    string name = CleanString(Console.ReadLine());

                    path = "Networks/";

                    Console.Out.WriteLine("Saving Network to Path: " + Directory.GetCurrentDirectory() + "/" + path);
                    string nnJson = nn.ToJson();

                    if (!Directory.Exists(path))
                        Directory.CreateDirectory(path);

                    File.WriteAllText(path + name + ".json", nn.ToJson());
                    break;
                case "n":
                    break;
            }
        }
        public static int ReadBigInt32(this BinaryReader br) {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian)
                Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        public static string CleanString(string dirtyString) {
            return new String(dirtyString.Where(Char.IsLetterOrDigit).ToArray());
        }
        private static void drawTextProgressBar(int progress, int total) {
            //draw empty progress bar
            Console.CursorLeft = 0;
            Console.Write("["); //start
            Console.CursorLeft = 32;
            Console.Write("]"); //end
            Console.CursorLeft = 1;
            float onechunk = 30.0f / total;

            //draw filled part
            int position = 1;
            for (int i = 0; i < onechunk * progress; i++) {
                Console.BackgroundColor = ConsoleColor.Gray;
                Console.CursorLeft = position++;
                Console.Write(" ");
            }

            //draw unfilled part
            for (int i = position; i <= 31; i++) {
                Console.BackgroundColor = ConsoleColor.Green;
                Console.CursorLeft = position++;
                Console.Write(" ");
            }

            //draw totals
            Console.CursorLeft = 35;
            Console.BackgroundColor = ConsoleColor.Black;
            Console.Write(progress.ToString() + " of " + total.ToString() + "    "); //blanks at the end remove any excess
        }

    }

}
