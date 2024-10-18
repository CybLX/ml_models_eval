from pylatex import Document, PageStyle, LineBreak,LargeText,MediumText,Tabu,Head,MiniPage,LongTable,Figure,StandAloneGraphic
from pylatex.utils import bold, NoEscape
from datetime import date
import json
import os

class Metrics_LaTeX ():
    def __init__(self):
        #---Initialize---
        geometry_options = {
                "head": "2.0cm",
                "left": "0.5cm",
                "right": "0.5cm",
                "bottom": "2.5cm"
                }
        self.doc = Document(geometry_options = geometry_options, inputenc = 'utf8')
    
    @staticmethod
    def Read(filename):
        with open(filename) as f:
            file = json.loads(f.read())
        return file

    def Read_Generate(self):

        metrics = self.Read(filename = './report/metrics.json')
        
        paths = list(metrics.keys())[8:]
        for path in paths:

            first_page = PageStyle("firstpage")

            self.Create(first_page = first_page,
                        name = path,
                        BD_frac = metrics[path]['BigData_fraction'],
                        class_rp = metrics[path]["class_rp"],
                        score = metrics[path]["score_"],
                        extime = metrics[path]["time_train"],
                        kappa_ = metrics[path]["kappa_"],
                        Roc_AUC = metrics[path]["lr_auc"],
                        f1_auc = metrics[path]["f1_auc"], 
                        proportion = metrics["proportion"],
                        analyser = metrics["analyzer"],
                        method = metrics['method'],
                        ngram = metrics["Ngrams"],                        
                        freq_threshold = metrics["freq_threshold"],
                        stop_word = metrics["stop_word"],
                        class_together = metrics["class_together"])
            self.doc.append(first_page) 
            self.doc.change_document_style("firstpage")
            self.doc.append(NoEscape(r"\newpage"))
        self.doc.generate_pdf(filepath = f'./Report_learning_{date.today()}', clean_tex = False, compiler = 'pdflatex')
        
        name = f'/Report_learning_{date.today()}'        
        os.rename("." + name + ".tex", "./report" + name + ".tex")
        os.rename("." + name + ".pdf", "./report" + name + ".pdf")

    @staticmethod
    def Create(first_page,name,BD_frac, class_rp,score,extime,kappa_,Roc_AUC,analyser,method,proportion,ngram,f1_auc,freq_threshold,stop_word,class_together):
        
        with first_page.create(Head("C")) as header:
            header.append(LargeText(bold(f"{name}")))
            header.append(LineBreak())
            header.append(MediumText(NoEscape(r"\textit{\today}")))

        with first_page.create(MiniPage(width=r"0.95\textwidth", pos = 'c')) as mini:

            with mini.create(MiniPage(width = r"0.45\textwidth")) as mini2:
                with mini2.create(LongTable("|l | l|", pos = 'c')) as first_page_table:
                        first_page_table.add_hline()
                        row1 = [bold('Score: '), f"{score}%"]
                        row2 = [bold("Execution Time: "), f"{round(extime,3)}s"]
                        row3 = [bold("BigData Fraction: "), f"{BD_frac*100}%"]
                        row4 = [bold("Cohen's kappa: "), f"{kappa_}%"]
                        row5 = [bold('Roc AUC: '), f"{round(Roc_AUC,3)}"]
                        row6 = [bold('Analyser: '), f"{analyser}"]
                        row7 = [bold('Analyser: '), f"{method}"]
                        row8 = [bold('Train Proprotion: '), f"{proportion*100}%"]
                        row9 = [bold('Ngrans: '), f"{ngram}"]
                        row10 = [bold("FreqThreshold: "), f"{freq_threshold}"]
                        row11 = [bold("StopWord: "), f"{stop_word}"]
                        row12 = [bold('OutTogether: '), f"{class_together}"]
                        

                        first_page_table.add_row(row1)
                        first_page_table.add_hline()
                        first_page_table.add_row(row2)
                        first_page_table.add_hline()
                        first_page_table.add_row(row3)
                        first_page_table.add_hline()
                        first_page_table.add_row(row4)
                        first_page_table.add_hline()
                        first_page_table.add_row(row5)
                        first_page_table.add_hline()
                        first_page_table.add_row(row6)
                        first_page_table.add_hline()
                        first_page_table.add_row(row7)
                        first_page_table.add_hline()
                        first_page_table.add_row(row8)
                        first_page_table.add_hline()
                        first_page_table.add_row(row9)
                        first_page_table.add_hline()
                        first_page_table.add_row(row10)
                        first_page_table.add_hline()
                        first_page_table.add_row(row11)
                        first_page_table.add_hline()
                        first_page_table.add_row(row12)
                        first_page_table.add_hline()

                                                
            with mini.create(MiniPage(width=r"0.55\textwidth")) as mini1:
                
                with mini1.create(Tabu("X[r] | X[r] X[r] X[r] X[r]", pos = 'b')) as data_table:
                    

                        data_table.add_hline()
                        data_table.add_row(["   ","precision", "recall", "f1-score", "support"],mapper=[bold])
                        data_table.add_hline()

                        suport_total = class_rp["positive"]['support'] + class_rp["neutral"]['support'] + class_rp["negative"]['support']
                        row = ["Negative",round(class_rp["negative"]['precision'],3), round(class_rp["negative"]['recall'],3), round(class_rp["negative"]['f1-score'],3),class_rp["negative"]['support']]
                        row2 = ['Neutral',round(class_rp["neutral"]['precision'],3),round(class_rp["neutral"]['recall'],3),round(class_rp["neutral"]['f1-score'],3),class_rp["neutral"]['support']]
                        row3 = ["Positive",round(class_rp["positive"]['precision'],3),round(class_rp["positive"]['recall'],3), round(class_rp["positive"]['f1-score'],3),class_rp["positive"]['support']]
                        row4 = ['Accuracy',"","",round(class_rp["accuracy"],3),suport_total]
                        row5 = ["Macro avg",round(class_rp["macro avg"]['precision'],3),round(class_rp["macro avg"]['recall'],3),round(class_rp["macro avg"]['f1-score'],3),suport_total]
                        row6 = ['Weighted avg',round(class_rp["weighted avg"]['precision'],3),round(class_rp["weighted avg"]['recall'],3),round(class_rp["weighted avg"]['f1-score'],3),suport_total]

                        data_table.add_row(row)
                        data_table.add_row(row2)
                        data_table.add_row(row3)
                        data_table.add_empty_row()
                        data_table.add_row(row4)
                        data_table.add_row(row5)
                        data_table.add_row(row6)

        
        first_page.append(LineBreak())
        with first_page.create(MiniPage(width= (r'0.7\textwidth'))) as mini:
            
            with mini.create(MiniPage(width = r"0.8\textwidth")) as mini2:
                mini2.append(StandAloneGraphic(image_options="width=360px",
                                    filename=f"./report/Learning/{name}/confusion-matrix.png"))
                        
            with mini.create(MiniPage(width=r"0.5\textwidth")) as mini1:
                with mini1.create(Tabu("X[r] | X[r] X[r]", pos = 't')) as data_table:
                    indexs,f1,aucs = f1_auc
                    data_table.add_hline()
                    header_row1 = ["Index", "F1", "AUC"]
                    data_table.add_row(header_row1, mapper=[bold])
                    data_table.add_hline()
                    data_table.add_row([indexs[0],round(f1[0],3),round(aucs[0],3)])
                    data_table.add_row([indexs[1],round(f1[1],3),round(aucs[1],3)])
                    data_table.add_row([indexs[2],round(f1[2],3),round(aucs[2],3)])


        with first_page.create(Figure(position='h!')) as kitten_pic:
           kitten_pic.add_image(f"./report/Learning/{name}/True-Positive-Rate.png", width='350px')
           kitten_pic.add_caption('Roc_auc_curve')
        first_page.append(NoEscape(r"\newpage"))

        first_page.append(NoEscape(r'\raggedright'))
        with first_page.create(Figure(position='h!')) as kitten_pic:
           kitten_pic.add_image(f"./report/Learning/{name}/Recall-Precision.png", width='365px')#, placement = NoEscape(r'\raggedright'))
           kitten_pic.add_caption('Precision_recall_curve')
