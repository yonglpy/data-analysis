options(warn = 0)
library(shiny)
library(shinydashboard)
library(VennDiagram)
library(pheatmap)
library(FactoMineR)
library(factoextra)
library(DESeq)
library(DESeq2)
library(genefilter)
library(org.Hs.eg.db)
library(clusterProfiler)
library(ggplot2)
library(DOSE)
library(DT)
library(pathview)
options(shiny.maxRequestSize=50*1024^2)
options(shiny.fullstacktrace=TRUE)
options(stringsAsFactors = FALSE)
options(expressions=12000)


if(TRUE){
  header <- dashboardHeader(title = "DM_demo",
                            dropdownMenu(type = "messages",
                                         messageItem(
                                           from = 'Sales Dept',
                                           message = 'Sales are steady this moth'
                                         ),
                                         messageItem(
                                           from = "New User",
                                           message = "how do I register?",
                                           icon = icon('question'),
                                           time='11:04'
                                         ),
                                         messageItem(
                                           from = 'Support',
                                           message = 'The new server is ready',
                                           icon = icon("life-ring"),
                                           time = '2018-10-23'
                                         )
                            ),
                            dropdownMenu(type = 'notifications',
                                         notificationItem(
                                           text = '5 new users today',
                                           icon("users")
                                         ),
                                         notificationItem(
                                           text = '12 items delivered',
                                           icon('truck'),
                                           status = 'success'
                                         ),
                                         notificationItem(
                                           text = 'Server load at 76%',
                                           icon = icon('exclamation-triangle'),
                                           status = 'warning'
                                         )
                            ),
                            dropdownMenu(type = 'tasks',badgeStatus = 'success',
                                         taskItem(value = 90,color = 'green',
                                                  "Documentation"
                                                  ),
                                         taskItem(value = 17,color = 'aqua',
                                                  'Documentation'
                                                  ),
                                         taskItem(value = 75,color = 'yellow',
                                                  'Documentation'
                                                  ),
                                         taskItem(value = 80,color = 'red',
                                                  "Overall project"
                                                  )
                            )
  )
}

if (TRUE){
  sidebar = dashboardSidebar(
    tags$head(tags$script(src="custom.js")),
    tags$head(tags$link(rel="stylesheet", type="text/css", href="custom.css")),
    sidebarMenu(
      menuItem("Metabolitics", tabName="metabolome", icon=icon("metabolome")),
      menuItem("Transcriptom", tabName = "transcriptom", icon=icon("transcriptom"),
        menuSubItem('Primary analysis',tabName = 'Primary'),
        menuSubItem('Advanced analysis',tabName = 'Advanced')
      ),
      menuItem("Metagenomics", tabName = "metagenome", icon=icon("transcriptom")),
      menuItem("Immunomics", tabName = "Immune", icon=icon("transcriptom"))
    )
  )
}
boxTitle = function(title="Inputs"){
  tags$div(class="boxtitle", title,
           HTML('<div style="display:inline;float:right"><i class="fa fa-window-minimize"></i></div>'),
           HTML('<div style="display:inline;float:right"><i class="fa fa-window-maximize"></i></div>'))
}
if (TRUE){
  Basic = tabItem(tabName = 'Primary',
                fluidRow(
                  box(width = 12,height = 400,status = 'primary',solidHeader = TRUE,
                      boxTitle('Data Input'),
                      box(width = 6,height = 340,title = 'Input 1:Gene expression matrix',status = 'primary',solidHeader = TRUE,
                          'Upload your datafile here,1st row contains sample name,
                          1st column is Gene name or Gene ID',
                          fileInput('data',multiple = FALSE,label = NULL,
                                    accept = c('.csv','.txt'),placeholder = 'default is data.csv'),
                          radioButtons('seq_data',label = 'Separator',choices = c(Comma =',',Semicolon = ':',Tab = '\t')),
                          checkboxInput('logged',label = 'data has been logged transform?',value = FALSE)
                          ),
                      box(width = 6,height = 340,title = 'Input 2:Gene reads counts file',status = 'primary',solidHeader = TRUE,
                          'Upload your gene readsCounts file,1st row contains sample name,
                          lst row is Gene name or Gene ID',
                          fileInput('ReadsCounts',multiple = FALSE,label = NULL,
                                    accept = c('.csv','.txt'),placeholder = 'default is data.csv'),
                          radioButtons('seq_ReadsCounts',label = 'Separator',choices = c(Comma = ',',Semicolon = ':',Tab = '\t'))
                          )
                      ),
                  box(width = 7,height = 600,status = 'primary',solidHeader = TRUE,
                      boxTitle('Box Plot'),
                      box(width = 3,title = 'Input 1:sample name',status = 'primary',solidHeader = TRUE,
                          textInput('samplename','Sample name',value = 'All')),
                      box(width = 9,title = 'Box plot',status = 'primary',solidHeader = TRUE,
                          plotOutput('boxplot'))
                  ),
                  box(width = 5,height = 600,status = 'primary',solidHeader = TRUE,
                      boxTitle('VennDigram'),
                      box(width = 5,title = 'select venn number',status = 'primary',solidHeader = TRUE,
                          selectInput('venn','venn number:',c('three','four','five')),
                          conditionalPanel(
                            condition = "input.venn == 'three'",
                            textInput('sample1','Sample:','X1T:X1P:X1N')
                          ),
                          conditionalPanel(
                            condition = "input.venn == 'four'",
                            textInput('sample2','Sample:','X1T:X1P:X1N:X2T')
                          ),
                          conditionalPanel(
                            condition = "input.venn == 'five'",
                            textInput('sample3','Sample:','X1T:X1P:X1N:X2T:X2N')
                          )
                        ),
                      box(width = 7,title = 'VennDigram plot',status = 'primary',solidHeader = TRUE,
                        plotOutput('venn')
                      )
                ),
                box(width = 12,height = 600,status = 'primary',solidHeader = TRUE,
                    boxTitle('HeatMap'),
                    box(width = 5,title = 'Heatmap parameter selection',status = 'primary',solidHeader = TRUE,
                        selectInput('correlation_method','Correlation method:',c('spearman','pearson')),
                        checkboxInput('cluster_row','Cluster Row:',value=TRUE),
                        checkboxInput('cluster_col','Cluster Col:',value=TRUE),
                        checkboxInput('display_number','Display number:',value = TRUE),
                        textInput('col1','Color:','red'),
                        textInput('col2','Color:','green'),
                        textInput('main','Main:','pheatmap')
                        ),
                    box(width = 7,title = 'Heatmap plot',status = 'primary',solidHeader = TRUE,
                        plotOutput('heatmap')
                        )
                ),
                box(width = 5,height = 600,status = 'primary',solidHeader = TRUE,
                    boxTitle('samples cluster'),
                    box(width = 5,title = 'Cluster parameter selection',status = 'primary',solidHeader = TRUE,
                      selectInput('cluster_method','clsuster method selection',c('ward.D','ward.D2','single','complete',
                                                                                 'average','mcquitty','median','centroid')),
                      numericInput("obs", "Cluster categories:",value=3),
                      textInput('xlab','xlab information','Sample Cluster')
                    ),
                    box(width = 7,title = 'Sample Cluster Plot',status = 'primary',solidHeader = TRUE,
                        plotOutput('cluster')
                    )
                ),
                box(width = 7,height = 600,status = 'primary',solidHeader = TRUE,
                    boxTitle('PCA Analysis'),
                    box(width = 3,title = 'PCA Analysis Parameter',status = 'primary',solidHeader = TRUE,
                       sliderInput('pch','Choice pch type',min = 1,max = 25,value = 20),
                       textInput('low_cor','low colour','red'),
                       textInput('medium','Medium colour','green'),
                       textInput('heigh','Heigh colour','black')
                    ),
                    box(width = 3,title = 'Percentage of variance',status = 'primary',solidHeader = TRUE,
                        plotOutput('variance')
                    ),
                    box(width = 6,title = 'PCA Plot',status = 'primary',solidHeader = TRUE,
                        plotOutput('PCA')
                    )
                )
            )
  )
}

if (TRUE){
  advanced = tabItem(tabName ='Advanced',
                     fluidRow(
                       box(width = 5,height = 600,status = 'primary',solidHeader = TRUE,
                           boxTitle('Data Input'),
                           box(width = 6,height = 500,title = 'Input:Gene reads counts file',status = 'primary',solidHeader = TRUE,
                               'Upload your gene readsCounts file,1st col contains sample name,
                               lst row is Gene name or Gene ID',
                               fileInput('ReadsCounts1',multiple = FALSE,label = NULL,accept = c('.csv','.txt'),
                                         placeholder = 'default is data.csv'),
                               radioButtons('seq_ReadsCounts1',label = 'Separator',choices = c(Comma = ',',Semicolon = ':',Tab = '\t'))
                           ),
                           box(width = 6,height = 500,title = 'Input:difference information',status = 'primary',solidHeader = TRUE,
                               'Upload your sample group information',
                               selectInput('Difference_method','Difference analysis method:',c('DESeq','DESeq2'),'DESeq'),
                               conditionalPanel(
                                 condition = "input.Difference_method == 'DESeq'",
                                 textInput('control','control sample name',value ='HP_HFp3'),
                                 textInput('deal','deal sample name',value = 'MI_HFp3')
                              ),
                              conditionalPanel(
                                condition = "input.Difference_method == 'DESeq2'",
                                textInput('control_Group','control group sample names',value = 'HP_HFp3;HP_HFm1;HP_HFr1'),
                                textInput('deal_Group','deal group sample names',value =  'MI_HFp3;MI_HFm2;MI_nHF1')
                              )
                           )
                       ),
                       box(width = 7,height = 600,status = 'primary',solidHeader = TRUE,
                           boxTitle('Difference gene expression volcano plot'),
                           box(width = 3,height = 500,title = 'Volcano parameter choice',status = 'primary',solidHeader = TRUE,
                                sliderInput('log2','log2 fold change',min = 0,max = 5,value = 1),
                                sliderInput('pvalue','p-value',min=0.01,max=0.1,value=0.05),
                                textInput('upcr','Up gene color',value = 'red'),
                                textInput('downcr','down gene color',value = 'blue')
                            ),
                           box(width = 9,height = 500,title = 'Volcano plot of differnece expression gene',status = 'primary',solidHeader = TRUE,
                               plotOutput('volcano')
                           )
                      ),
                      box(width = 6,height = 600,status = 'primary',solidHeader = TRUE,
                          boxTitle('Difference expression gene cluster analysis'),
                          box(width = 4,height = 500,title = 'Cluster parameter choices',status = 'primary',solidHeader = TRUE,
                              checkboxInput('diff_cluter_row','difference gene cluster by row',value = TRUE),
                              checkboxInput('sample_cluster_col','sample cluster by col',value = FALSE ),
                              textInput('UP_color','gene up expression color','red'),
                              textInput('Down_color','gene down expression color','green'),
                              checkboxInput('show_rownames','show rownames',value = FALSE),
                              checkboxInput('show_colnames','show colnames',value = TRUE)
                           ),
                          box(width = 8,height = 500,title = 'difference expression gene heatmap',status = 'primary',solidHeader = TRUE,
                              plotOutput('diffheatmap')
                          )
                      ),
                      box(width = 6,height = 600,status = 'primary',solidHeader = TRUE,
                          boxTitle('KEGG Pathway enrichment'),
                          box(width = 4,height = 500,title = 'KEGG Pathway enrichment parameter',status = 'primary',solidHeader = TRUE,
                              selectInput('GeneinputType','input Gene id Type',c('ENSEMBL','ENTREZID','SYMBOL')),
                              sliderInput('pvaluecutoff','p-value cutoff value',min=0,max= 1,value = 0.5),
                              selectInput('enrichment_method','Kegg pathway enrichment method',c('BH','holm','hochberg','"bonferroni','fdr')),
                              sliderInput('qcutoff','qvalue cutoff value',min = 0,max = 1,value = 0.4),
                              numericInput('enrichmentnumber','KEGG pathway number will plot',value = 10,min = 1, max = 100)
                          ),
                          box(width = 8,height = 500,title = 'KEGG Pathway enrichment plot',status = 'primary',solidHeader = TRUE,
                              plotOutput('Keggenrichment')
                          )
                      ),
                      box(width = 5,height = 600,status = 'primary',solidHeader = TRUE,footer = NULL,style="overflow:auto",
                          boxTitle('KEGG enrichment result information'),
                          DT::dataTableOutput('table',width = '100%',height = 'auto')
                      ),
                      box(width = 7,height = 600,status = 'primary',solidHeader = TRUE,
                          boxTitle('show pathway'),
                          box(width = 3,height = 500,title = 'input pathway id',status = 'primary',solidHeader = TRUE,
                              textInput('PathwayID','input pathway ID','hsa05143')
                              ),
                          box(width = 9,height = 500,title = 'pathway full view',status = 'primary',solidHeader = TRUE,style="overflow:auto",
                            imageOutput('pathway')
                            )
                      ),
                      box(width = 12,height = 600,status = 'primary',solidHeader = TRUE,
                          boxTitle('GO Enrichment'),
                          box(width = 3,height = 540,title = 'GO Enrichment parameter',status = 'primary',solidHeader = TRUE,
                              selectInput('GOinputGeneIDType','GO Enrichment Gene ID Type',c('ENSEMBL','ENTREZID','SYMBOL')),
                              sliderInput('GP','GO enrichment p-value cutoff value',min=0,max= 1,value = 0.5),
                              sliderInput('GQ','GO enrichment q-value cutoff value',min = 0,max = 1,value = 0.4),
                              numericInput('GN','GO item number will plot',value=10,min=1,max=100),
                              selectInput('GO_method','GO enrichment method',c('BH','holm','hochberg','"bonferroni','fdr')),
                              selectInput('GOType','GO Ontologies',c('BP','MF','CC'))
                             ),
                          box(width = 5,height = 540,title = 'GO Enrichment dotPlot',status = 'primary',solidHeader = TRUE,
                              plotOutput('GOenrichment')
                             ),
                          box(width = 4,height = 540,title = 'GO Enrichment barplot',status = 'primary',solidHeader = TRUE,
                              plotOutput('GObarplot')
                             )
                      ),
                      box(width = 12,height = 700,status = 'primary',solidHeader = TRUE,
                          boxTitle('Plot GO Graph'),
                          box(width = 8,height = 660,title = 'GO Graph plot',status = 'primary',solidHeader = TRUE,style="overflow:auto",
                              plotOutput('GOgraph')
                              )
                      )
            )
  )
}
if (TRUE){
  met = tabItem(tabName="metabolome",
                box(
                  h1('luan en hui')
                ))
}
if (TRUE){
  metagenome = tabItem(tabName="metagenome",
                       box(h1("ding qiuxia")))
}
if (TRUE){
  immune = tabItem(tabName="Immune",
                   box(h1("Yan MingChen")))
}

body = dashboardBody(
  tabItems(met,Basic,advanced,metagenome,immune)
)

ui<-dashboardPage(header,sidebar,body)
server <- function(input,output,session){
  dataFile = reactive({
    if (is.null(input$data)){
      read.csv('D:\\lenovo\\shiny\\all_samples_gene_TPM.csv',header = TRUE,sep = ',',stringsAsFactors = FALSE)
    }
    else {
      read.csv(input$data$datapath,header = TRUE,sep = input$seq_data,stringsAsFactors = FALSE)
    }
  })
  dataRC = reactive({
    if (is.null(input$ReadsCounts1)){
      read.csv('D:\\lenovo\\shiny\\Difference\\all_sample_gene_readscounts.txt',header = TRUE,sep='\t',stringsAsFactors = TRUE,row.names = 'GeneID')
    } else {
      read.csv(input$ReadsCounts1$ReadsCounts1path,header = TRUE,sep = input$seq_ReadsCounts1,stringsAsFactors = FALSE,row.names = 1)
    }
  })
  output$boxplot <- renderPlot({
    sampleid <- reactive({
      if (input$samplename == 'All'){
        input$samplename
      } else {
        paste('X',input$samplename,sep='')
      }
    })
    sample_data <- reactive({
      if (input$samplename == 'All'){
        dataFile()
      } else {
        dataFile()[,sampleid()]
      }
    })
    data1 <- reactive({
      log2(sample_data()+1)
    })
    name1 <- reactive({
      if (sampleid() == 'All'){
        gsub('X','',colnames(dataFile()))
      } else {
        gsub('X','',sampleid())
      }
    })
    if (sampleid() == 'All'){
      boxplot(data1(),names=name1(),xlab='sample name',ylab='log2(TPM + 1)',col=rainbow(length(name1())))
    } else {
      boxplot(data1(),main=name1(),xlab=paste('sample name','',name1()),ylab='log2(TPM+1)',col=rainbow(length(name1())))
    }
  })
  output$venn <- renderPlot({
    if(input$venn == 'three') {
      samplelist = reactive({
        strsplit(input$sample1,split=':')[[1]]
      })
      #print(isolate(samplelist()))
      samplename3 <- reactive({
        gsub('X','',samplelist())
      })
      sample1 <- reactive({
        a = subset(dataFile(),select=c(samplelist()[1]))
        row.names(subset(a,a>0))
      })
      sample2 <- reactive({
        b = subset(dataFile(),select=c(samplelist()[2]))
        row.names((subset(b,b>0)))
      })
      sample3 <- reactive({
        c=subset(dataFile(),select=c(samplelist()[3]))
        row.names(subset(c,c>0))
      })
      VD <- venn.diagram(x = list(T=sample1(),P=sample2(),N=sample3()),filename=NULL,fill=rainbow(3),
                         category.names = c(samplename3()[1],samplename3()[2],samplename3()[3]))
      grid.draw(VD)
    }
    if(input$venn == 'four'){
      samplelist = reactive({
        strsplit(input$sample2,split=':')[[1]]
      })
      samplename3 <- reactive({
        gsub('X','',samplelist())
      })
      sample1 <- reactive({
        a = subset(dataFile(),select=c(samplelist()[1]))
        row.names(subset(a,a>0))
      })
      sample2 <- reactive({
        b = subset(dataFile(),select=c(samplelist()[2]))
        row.names((subset(b,b>0)))
      })
      sample3 <- reactive({
        c=subset(dataFile(),select=c(samplelist()[3]))
        row.names(subset(c,c>0))
      })
      sample4 <- reactive({
        d=subset(dataFile(),select=c(samplelist()[4]))
        row.names(subset(d,d>0))
      })
      VD <- venn.diagram(x = list(T=sample1(),P=sample2(),N=sample3(),M=sample4()),filename=NULL,fill=rainbow(4),
                         category.names = c(samplename3()[1],samplename3()[2],samplename3()[3],samplename3()[4]))
      grid.draw(VD)
    }
    if (input$venn == 'five'){
      samplelist = reactive({
        strsplit(input$sample3,split=':')[[1]]
      })
      samplename3 <- reactive({
        gsub('X','',samplelist())
      })
      sample1 <- reactive({
        a = subset(dataFile(),select=c(samplelist()[1]))
        row.names(subset(a,a>0))
      })
      sample2 <- reactive({
        b = subset(dataFile(),select=c(samplelist()[2]))
        row.names((subset(b,b>0)))
      })
      sample3 <- reactive({
        c=subset(dataFile(),select=c(samplelist()[3]))
        row.names(subset(c,c>0))
      })
      sample4 <- reactive({
        d=subset(dataFile(),select=c(samplelist()[4]))
        row.names(subset(d,d>0))
      })
      sample5 <- reactive({
        e=subset(dataFile(),select=c(samplelist()[5]))
        row.names(subset(e,e>0))
      })
      VD <- venn.diagram(x = list(T=sample1(),P=sample2(),N=sample3(),M=sample4(),O=sample5()),filename=NULL,fill=rainbow(5),
                         category.names = c(samplename3()[1],samplename3()[2],samplename3()[3],samplename3()[4],samplename3()[5]))
      grid.draw(VD)
    }
  })
  sample_name <- reactive({
    gsub('X','',colnames(dataFile()))
  })
  output$heatmap <- renderPlot({
    sample_cor <- reactive({
      cor(dataFile(),method = input$correlation_method)
    })
    pheatmap(sample_cor(),color = colorRampPalette(c(input$col1,input$col2))(100),cluster_cols = input$cluster_col,
             cluster_rows = input$cluster_row,display_numbers = input$display_number,labels_row = sample_name(),
             labels_col = sample_name(),width = 40,height = 40)
  })
  output$cluster <- renderPlot({
    cluster_data <- reactive({
      op = as.matrix(dataFile())
      colnames(op) = gsub('X','',colnames(op))
      t(op)
    })
    hc <- hclust(dist(cluster_data()),method = input$cluster_method,members = NULL)
    plot(hc,hang = -1,xlab =input$xlab)
    rect.hclust(hc,k=input$obs)
  })
  pca <- reactive({
    PCA(dataFile(),graph = FALSE)
  })
  output$variance <- renderPlot({
    fviz_eig(pca(),addlabels = TRUE)
  })
  var <-reactive({
    get_pca_var(pca())
  })
  output$PCA <- renderPlot({
    ko <- reactive({
      ko = pca()$eig[,2]
    })
    #print(isolate(ko()))
    xlab_name = reactive({
      paste('PCA1',"(",round(as.data.frame(ko())[,1][1],2),'%',")",sep = '')
    })
    ylab_name <- reactive({
      paste('PCA2',"(",round(as.data.frame(ko())[,1][2],2),"%",")",sep = '')
    })
    colour=colorRampPalette(c(input$low_cor,input$medium,input$heigh))(length(sample_name()))
    #par(mar=c(10,10,10,10))
    plot(var()$coord[,1:2],pch=input$pch,col=colour,xlab = xlab_name(),ylab = ylab_name())
    legend('top',legend = sample_name(),col=colour,pch=input$pch,bg='white',horiz = TRUE,xpd = TRUE,inset = -0.1,cex = 0.6)
  })
  samplediff_data <- reactive({
    if(input$Difference_method == 'DESeq'){
      dataRC()[,c(input$control,input$deal)]
    }
  })
  draw_volcano_data <- reactive({
    if(input$Difference_method == 'DESeq'){
      resAB <- reactive({
        control_name = input$control
        deal_name = input$deal
        conds = factor(c(control_name,deal_name))
        cds <- newCountDataSet(samplediff_data(),conds)
        cdsAB <- cds[,c(control_name,deal_name)]
        cdsAB <- estimateSizeFactors(cdsAB)
        cdsAB <- estimateDispersions(cdsAB,method="blind", fitType="local",sharingMode="fit-only")
        nbinomTest(cdsAB,control_name,deal_name)
      })
      a = na.omit(resAB())
      b = subset(a,log2FoldChange!=-Inf & log2FoldChange!=Inf)
      d=data.frame(GeneID=c(b$id),log2=c(b$log2FoldChange),FDR=c(b$padj))
      n = length(d$log2)
      l = c()
      p = c()
      GeneIDlist = c()
      updown = c()
      for(i in seq(1,n,1)){
        infor = d[i,]
        foldchange = infor$log2
        fdr = infor$FDR
        gid = infor$GeneID
        if(foldchange >= input$log2 & fdr <= input$pvalue){
          l[i]=foldchange
          p[i]=fdr
          updown[i]='Up'
          GeneIDlist[i] = gid
        }else if(foldchange <= -input$log2 & fdr <= input$pvalue){
          l[i]=foldchange
          p[i]=fdr
          updown[i]='Down'
          GeneIDlist[i] = gid
        } else {
          l[i]=foldchange
          p[i]=fdr
          updown[i]='*'
          GeneIDlist[i] = gid
        }
      }
      data.frame(GeneID=c(GeneIDlist),log2=c(l),FDR=c(p),UD=c(updown))
    }
  })
  groupdiffinfor = reactive({
    if(input$Difference_method == 'DESeq2'){
      c(c(input$control_Group),c(input$deal_Group))
    }
  })
  groupdiff_data <- reactive({
    if(input$Difference_method == 'DESeq2'){
      control_sample_list = strsplit(groupdiffinfor()[1],split = ';')[[1]]
      deal_sample_list = strsplit(groupdiffinfor()[2],split = ';')[[1]]
      dataRC()[,c(control_sample_list[1],control_sample_list[2],control_sample_list[3],deal_sample_list[1],deal_sample_list[2],deal_sample_list[3])]
    }
  })
  draw_volcano_data_group<- reactive({
    if(input$Difference_method == 'DESeq2'){
      resAB = reactive({
        group_number = length(strsplit(groupdiffinfor()[1],split = ';')[[1]])
        condition <- factor(c(rep("control",group_number), rep("treat",group_number)))
        coldata <- data.frame(row.names = colnames(groupdiff_data()), condition)
        dds <- DESeqDataSetFromMatrix(countData=groupdiff_data(), colData=coldata, design=~condition)
        dds <- DESeq(dds,quiet=TRUE)
        res <- results(dds, cooksCutoff=FALSE, independentFiltering=FALSE, pAdjustMethod="BH")
        merge(as.data.frame(res), as.data.frame(counts(dds, normalized=TRUE)),by="row.names",sort=FALSE)
      })
      grdata = na.omit(resAB())
      grdata_filter = subset(grdata,log2FoldChange!=Inf & log2FoldChange!=Inf)
      grdata_frame = data.frame(GeneID=c(grdata_filter$Row.names),log2=grdata_filter$log2FoldChange,FDR=grdata_filter$padj)
      n = length(grdata_frame$log2)
      l=c()
      p=c()
      updown=c()
      GeneIDlist=c()
      for(i in seq(1,n,1)){
        infor = grdata_frame[i,]
        foldchange = infor$log2
        fdr = infor$FDR
        gid = infor$GeneID
        if(foldchange >= input$log2 & fdr <= input$pvalue){
          l[i] = foldchange
          p[i] = fdr
          updown[i]='Up'
          GeneIDlist[i] = gid
        }else if(foldchange <= -input$log2 & fdr <= input$pvalue){
          l[i]=foldchange
          p[i]=fdr
          updown[i]='Down'
          GeneIDlist[i] = gid
        }else{
          l[i] = foldchange
          p[i] = fdr
          updown[i] = '*'
          GeneIDlist[i] = gid
        }
      }
      data.frame(GeneID=c(GeneIDlist),log2=c(l),FDR=c(p),UD=c(updown))
    }
  })
  output$volcano <- renderPlot({
    if(input$Difference_method == 'DESeq'){
      no_degs_x <- draw_volcano_data()$log2[ grepl('\\*',draw_volcano_data()$UD,perl=TRUE) ]
      no_degs_y <- draw_volcano_data()$FDR[ grepl('\\*',draw_volcano_data()$UD,perl=TRUE) ]
      degs_x <- draw_volcano_data()$log2[ grepl("[^\\*]",draw_volcano_data()$UD,perl=TRUE) ]
      degs_y <- draw_volcano_data()$FDR[ grepl("[^\\*]",draw_volcano_data()$UD,perl=TRUE) ]
      no_degs_y <- -log10(no_degs_y)
      degs_y <- -log10(degs_y)
      aa <- factor(draw_volcano_data()$UD)
      number <- as.data.frame(table(aa))
      xmin <- min(c(no_degs_x,degs_x)[grepl("\\d+",c(no_degs_x,degs_x),perl=TRUE)])
      xmax <- max(c(no_degs_x,degs_x)[grepl("\\d+",c(no_degs_x,degs_x),perl=TRUE)])
      ymin <- min(c(no_degs_y,degs_y)[grepl("\\d+",c(no_degs_y,degs_y),perl=TRUE)])
      ymax <- max(c(no_degs_y,degs_y)[grepl("\\d+",c(no_degs_y,degs_y),perl=TRUE)])
      par(mar=c(5,5,5,10)+0.1,xpd=TRUE)
      plot(0,0,pch="",xlim=c(xmin,xmax),ylim=c(ymin,ymax),ylab='-log10(q-value)',xlab='log2(fold change)',
           cex.lab = 1.5,main="Volcano plot for DESeq method",cex=1)
      for (i in 1 : length(no_degs_x)) {
        points(no_degs_x[i], no_degs_y[i], col = "grey65", pch = 20, cex = 0.3)
      }
      for (j in 1 : length(degs_x)) {
        if (degs_x[j] > 0) {
          points(degs_x[j], degs_y[j], col = input$upcr, pch = 20, cex = 0.3)
        }else{
          points(degs_x[j], degs_y[j], col = input$downcr, pch = 20, cex = 0.3)
        }
      }
      legend(xmax+(xmax-xmin)/12,(ymax+ymin)/2+(ymax-ymin)/5-(ymax-ymin)/15, 
             legend=expression(atop(italic("log")[2]*FoldChange >=1.~ italic(",FDR") <=0.05)), pch="",bty = "n",pt.cex=1,cex=0.5)
      legend(xmax+(xmax-xmin)/12,(ymax+ymin)/2-(ymax-ymin)/15, legend=expression(atop(italic("log")[2]*FoldChange <=-1.~ italic(",FDR") <=0.05)), 
             pch="",bty = "n", pt.cex=1.8, cex=0.5)
      legend(xmax+(xmax-xmin)/12,(ymin+ymax)/2-(ymax-ymin)/5-(ymax-ymin)/15, legend=expression(atop(italic("abs(log")[2]*"FoldChange)" <1.~ italic(" or FDR") >0.05)), 
             pch="",bty = "n", pt.cex=1.8, cex=0.5)
      legend(xmax+(xmax-xmin)/15,(ymax+ymin)/2+(ymax-ymin)/5, legend=paste("Up:",number[3,2]),col=input$upcr,pch=20,bty = "n", pt.cex=2, cex=0.5)
      legend(xmax+(xmax-xmin)/15,(ymax+ymin)/2, legend=paste("Down:",number[2,2]),col=input$downcr,pch=20,bty = "n", pt.cex=2, cex=0.5)
      legend(xmax+(xmax-xmin)/15,(ymin+ymax)/2-(ymax-ymin)/5, legend=paste("no-DEGs:",number[1,2]),col="grey65",pch=20,bty = "n", pt.cex=2, cex=0.5)
    }
    if(input$Difference_method == 'DESeq2'){
      no_degs_x <- draw_volcano_data_group()$log2[ grepl('\\*',draw_volcano_data_group()$UD,perl=TRUE) ]
      no_degs_y <- draw_volcano_data_group()$FDR[ grepl('\\*',draw_volcano_data_group()$UD,perl=TRUE) ]
      degs_x <- draw_volcano_data_group()$log2[ grepl("[^\\*]",draw_volcano_data_group()$UD,perl=TRUE) ]
      degs_y <- draw_volcano_data_group()$FDR[ grepl("[^\\*]",draw_volcano_data_group()$UD,perl=TRUE) ]
      no_degs_y <- -log10(no_degs_y)
      degs_y <- -log10(degs_y)
      aa <- factor(draw_volcano_data_group()$UD)
      number <- as.data.frame(table(aa))
      xmin <- min(c(no_degs_x,degs_x)[grepl("\\d+",c(no_degs_x,degs_x),perl=TRUE)])
      xmax <- max(c(no_degs_x,degs_x)[grepl("\\d+",c(no_degs_x,degs_x),perl=TRUE)])
      ymin <- min(c(no_degs_y,degs_y)[grepl("\\d+",c(no_degs_y,degs_y),perl=TRUE)])
      ymax <- max(c(no_degs_y,degs_y)[grepl("\\d+",c(no_degs_y,degs_y),perl=TRUE)])
      par(mar=c(5,5,5,10)+0.1,xpd=TRUE)
      plot(0,0,pch="",xlim=c(xmin,xmax),ylim=c(ymin,ymax),ylab='-log10(q-value)',xlab='log2(fold change)',
           cex.lab = 1.5,main="Volcano plot for DESeq2 method",cex=1)
      for (i in 1 : length(no_degs_x)) {
        points(no_degs_x[i], no_degs_y[i], col = "grey65", pch = 20, cex = 0.3)
      }
      for (j in 1 : length(degs_x)) {
        if (degs_x[j] > 0) {
          points(degs_x[j], degs_y[j], col = input$upcr, pch = 20, cex = 0.3)
        }else{
          points(degs_x[j], degs_y[j], col = input$downcr, pch = 20, cex = 0.3)
        }
      }
      legend(xmax+(xmax-xmin)/12,(ymax+ymin)/2+(ymax-ymin)/5-(ymax-ymin)/15, 
             legend=expression(atop(italic("log")[2]*FoldChange >=1.~ italic(",FDR") <=0.05)), pch="",bty = "n",pt.cex=1,cex=0.5)
      legend(xmax+(xmax-xmin)/12,(ymax+ymin)/2-(ymax-ymin)/15, legend=expression(atop(italic("log")[2]*FoldChange <=-1.~ italic(",FDR") <=0.05)), 
             pch="",bty = "n", pt.cex=1.8, cex=0.5)
      legend(xmax+(xmax-xmin)/12,(ymin+ymax)/2-(ymax-ymin)/5-(ymax-ymin)/15, legend=expression(atop(italic("abs(log")[2]*"FoldChange)" <1.~ italic(" or FDR") >0.05)), 
             pch="",bty = "n", pt.cex=1.8, cex=0.5)
      legend(xmax+(xmax-xmin)/15,(ymax+ymin)/2+(ymax-ymin)/5, legend=paste("Up:",number[3,2]),col=input$upcr,pch=20,bty = "n", pt.cex=2, cex=0.5)
      legend(xmax+(xmax-xmin)/15,(ymax+ymin)/2, legend=paste("Down:",number[2,2]),col=input$downcr,pch=20,bty = "n", pt.cex=2, cex=0.5)
      legend(xmax+(xmax-xmin)/15,(ymin+ymax)/2-(ymax-ymin)/5, legend=paste("no-DEGs:",number[1,2]),col="grey65",pch=20,bty = "n", pt.cex=2, cex=0.5)
    }
  })
  output$diffheatmap <- renderPlot({
    if(input$Difference_method == 'DESeq'){
      gene=c()
      updown_list=c()
      control_readscounts=c()
      treat_readscounts=c()
      a_color=c()
      gene_id_list = subset(draw_volcano_data(),UD=='Up' | UD=='Down',select=c('GeneID'))$GeneID
      for(i in seq(1,length(gene_id_list),1)){
        infor = subset(draw_volcano_data(),GeneID == gene_id_list[i])
        if(infor$UD == 'Up'){
          a_color[i] = input$UP_color
        }
        if(infor$UD == 'Down'){
          a_color[i] = input$Down_color
        }
        if(samplediff_data()[gene_id_list[i],input$control]>0){
          control_readscounts[i] = log10(samplediff_data()[gene_id_list[i],input$control])
        }else{
          control_readscounts[i]=0.001
        }
        if(samplediff_data()[gene_id_list[i],input$deal]>0){
          treat_readscounts[i] = log10(samplediff_data()[gene_id_list[i],input$deal])
        }else{
          treat_readscounts[i] = 0.001
        }
        item = strsplit(gene_id_list[i],split='.',fixed = TRUE)[[1]][1]
        updown_list[i] = infor$UD
        gene[i] = item
      }
      sample_cluster_data = data.frame(GeneID=c(gene),Control=c(control_readscounts),Treat=c(treat_readscounts),row.names='GeneID')
      colnames(sample_cluster_data)=c(input$control,input$deal)
      id.par = data.frame(Gene=c(gene),UD=c(updown_list),col=c(a_color))
      annotation_row = data.frame(Gene = factor(id.par$UD))
      rownames(annotation_row) = id.par$Gene
      anno_color = list(Gene=c(Up=input$UP_color,Down=input$Down_color))
      pheatmap(sample_cluster_data , show_rownames=input$show_rownames , show_colnames=input$show_colnames , 
               cluster_rows=input$diff_cluter_row , cluster_cols=input$sample_cluster_col,
               annotation_row=annotation_row,annotation_colors=anno_color)
    }
    if(input$Difference_method == 'DESeq2'){
      grouplist = colnames(groupdiff_data())
      ggene_id_list = subset(draw_volcano_data_group(),UD=='Up' | UD=='Down',select=c('GeneID'))$GeneID
      grouplist_data = groupdiff_data()[ggene_id_list,grouplist]
      grouplist_data[grouplist_data==0] <- 0.01
      grouplist_data=log10(grouplist_data)
      ggene=c()
      gupdown_list=c()
      ga_color=c()
      for(j in seq(1,length(ggene_id_list),1)){
        ginfor = subset(draw_volcano_data_group(),GeneID == ggene_id_list[j])
        if(ginfor$UD == 'Up'){
          ga_color[j] = input$UP_color
        }
        if(ginfor$UD == 'Down'){
          ga_color[j] = input$Down_color
        }
        gitem = strsplit(ggene_id_list[j],split='.',fixed = TRUE)[[1]][1]
        ggene[j] = gitem
        gupdown_list[j] = ginfor$UD
      }
      rownames(grouplist_data)=ggene
      gid.par= data.frame(Gene=c(ggene),UD=c(gupdown_list),col=c(ga_color))
      gannotation_row = data.frame(Gene = factor(gid.par$UD))
      rownames(gannotation_row) = gid.par$Gene
      ganno_color = list(Gene=c(Up=input$UP_color,Down=input$Down_color))
      pheatmap(grouplist_data , show_rownames=input$show_rownames , show_colnames=input$show_colnames ,
               cluster_rows=input$diff_cluter_row , cluster_cols=input$sample_cluster_col,
               annotation_row=gannotation_row,annotation_colors=ganno_color)
    }
  })
  onesample_enrichment_result <- reactive({
    if(input$Difference_method == 'DESeq'){
      sample_enrichment_data = subset(draw_volcano_data(),UD=='Up' | UD=='Down',select=c('GeneID'))$GeneID
      Enrichment_Genelist = c()
      for(i in seq(1,length(sample_enrichment_data),1)){
        item = strsplit(sample_enrichment_data[i],split = '.',fixed = TRUE)[[1]][1]
        Enrichment_Genelist[i] = item
      }
      df <- bitr(Enrichment_Genelist,fromType = input$GeneinputType,toType = c('ENTREZID','SYMBOL'),OrgDb = org.Hs.eg.db,drop = TRUE)
      enrichKEGG(gene = df$ENTREZID,organism = 'hsa',keyType = 'kegg',pvalueCutoff = input$pvaluecutoff,
                pAdjustMethod = input$enrichment_method , minGSSize = 1,maxGSSize = 500,qvalueCutoff = input$qcutoff )
    }
  })
  group_enrichment_result <- reactive({
    if(input$Difference_method == 'DESeq2'){
      group_enrichment_data = subset(draw_volcano_data_group(),UD=='Up' | UD == 'Down',select=c('GeneID'))$GeneID
      group_enrichment_Genelist=c()
      for(i in seq(1,length(group_enrichment_data),1)){
        gitem = strsplit(group_enrichment_data[i],split = '.',fixed = TRUE)[[1]][1]
        group_enrichment_Genelist[i] = gitem
      }
      gdf <- bitr(group_enrichment_Genelist,fromType = input$GeneinputType,toType = c('ENTREZID','SYMBOL'),OrgDb = org.Hs.eg.db,drop = TRUE)
      enrichKEGG(gene = gdf$ENTREZID,organism = 'hsa',keyType = 'kegg',pvalueCutoff = input$pvaluecutoff,
                pAdjustMethod = input$enrichment_method ,minGSSize = 1,maxGSSize = 500,qvalueCutoff = input$qcutoff)
    }
  })
  output$Keggenrichment <- renderPlot({
    if(input$Difference_method == 'DESeq'){
      p <- DOSE::dotplot(onesample_enrichment_result(),showCategory=input$enrichmentnumber,font.size=10)
      show(p)
    }
    if(input$Difference_method == 'DESeq2'){
      p <- DOSE::dotplot(group_enrichment_result(),showCategory=input$enrichmentnumber,font.size=10)
      show(p)
    }
  })
  output$table <- renderDataTable({
    if(input$Difference_method == 'DESeq'){
      sample_data_frame = data.frame(onesample_enrichment_result())
      #print(head(sample_data_frame))
      adsd=sample_data_frame[,c('ID','Description','pvalue','qvalue','p.adjust','Count')]
      data.frame(ID=adsd$ID,Description=adsd$Description,pvalue=c(round(adsd$pvalue,4)),pvalue=c(round(adsd$pvalue,4)),Count=adsd$Count)
    } else {
      group_data_frame = data.frame(group_enrichment_result())
      ga = group_data_frame[,c('ID','Description','pvalue','qvalue','p.adjust','Count')]
      data.frame(ID=ga$ID,Description=ga$Description,pvalue=c(round(ga$pvalue,4)),pvalue=c(round(ga$pvalue,4)),Count=ga$Count)
    }
  })
  output$pathway <- renderImage({
    if(input$Difference_method == 'DESeq'){
      sample_pathway_data = subset(draw_volcano_data(),UD=='Up' | UD=='Down',select=c('GeneID','log2'))
      sample_pathway_geneList = c()
      for(i in seq(1,length(sample_pathway_data$GeneID),1)){
        item = strsplit(sample_pathway_data$GeneID[i],split='.',fixed = TRUE)[[1]][1]
        sample_pathway_geneList[i] = item
      }
      sample_pathway_data1 = data.frame(GeneID=sample_pathway_geneList,log2=sample_pathway_data$log2,row.names = 'GeneID')
      sample_df <- bitr(sample_pathway_geneList,fromType = input$GeneinputType,toType = c('ENTREZID','SYMBOL'),
                        OrgDb = org.Hs.eg.db,drop = TRUE)
      sample_pathway_log2=sample_pathway_data1[c(sample_df$ENSEMBL),]
      sample_pathway_data2 = as.matrix(data.frame(ENTREZID=sample_df$ENTREZID,log2=sample_pathway_log2,
                                                  row.names = 'ENTREZID'))
      width  <- session$clientData$output_pathway_width
      height <- session$clientData$output_pathway_height
      soutfile = paste('~/',input$PathwayID,'.sample.png',sep='')
      pathview(gene.data = sample_pathway_data2[,1],pathway.id = input$PathwayID,species = 'hsa',
               limit = list(gene=max(abs(sample_pathway_log2)),cpd=1),out.suffix='sample')
      list(src=soutfile,width='100%',height=400,alt=soutfile)
    } else{
      group_pathway_data = subset(draw_volcano_data_group(),UD=='Up' | UD=='Down',select=c('GeneID','log2'))
      group_path_geneList = c()
      for(j in seq(1,length(group_pathway_data$GeneID),1)){
        gitem = strsplit(group_pathway_data$GeneID[j],split = '.',fixed = TRUE)[[1]][1]
        group_path_geneList[j] = gitem
      }
      group_pathway_data1 = data.frame(GeneID=group_path_geneList,log2=group_pathway_data$log2,row.names = 'GeneID')
      group_df <- bitr(group_path_geneList,fromType = input$GeneinputType,toType = c('ENTREZID','SYMBOL'),
                       OrgDb = org.Hs.eg.db,drop = TRUE)
      #print(group_df)
      group_pathway_log2 = group_pathway_data1[c(group_df$ENSEMBL),]
      group_pathway_data2 = as.matrix(data.frame(ENTREZID=group_df$ENTREZID,log2=group_pathway_log2))
      width  <- session$clientData$output_pathway_width
      height <- session$clientData$output_pathway_height
      goutfile = paste('~/',input$PathwayID,'.group.png',sep='')
      pathview(gene.data =group_pathway_data2[,1],pathway.id = input$PathwayID,species = 'hsa',
               limit = list(gene=max(abs(group_pathway_log2)),cpd=1),out.suffix='group')
      list(src=goutfile,width='100%',height=400,alt=goutfile)
    }
  },deleteFile = TRUE)
  sample_GO_data <- reactive({
    if(input$Difference_method == 'DESeq'){
      a=subset(draw_volcano_data(),UD=='Up' | UD=='Down',select=c('GeneID'))$GeneID
      b=c()
      for(i in seq(1,length(a),1)){
        item = strsplit(a[i],split = '.',fixed = TRUE)[[1]][1]
        b[i] = item
      }
      df <- bitr(b,fromType = input$GOinputGeneIDType,toType = c('ENTREZID','SYMBOL'),OrgDb = org.Hs.eg.db,drop = TRUE)
      enrichGO(gene = df$ENTREZID,OrgDb = org.Hs.eg.db,keyType = 'ENTREZID',ont = input$GOType,pvalueCutoff = input$GP,
               pAdjustMethod = input$GO_method,qvalueCutoff = input$GQ,minGSSize = 1,maxGSSize = 400,readable = TRUE)
    }
  })
  group_GO_enrichment <- reactive({
    if(input$Difference_method == 'DESeq2'){
      group_GO_data = subset(draw_volcano_data_group(),UD=='Up' | UD == 'Down',select=c('GeneID'))$GeneID
      group_GO_genelist = c()
      for(j in seq(1,length(group_GO_data),1)){
        gitem = strsplit(group_GO_data[j],split = '.',fixed = TRUE)[[1]][1]
        group_GO_genelist[j] = gitem
      }
      gdf <- bitr(group_GO_genelist,fromType = input$GOinputGeneIDType,toType = c('ENTREZID','SYMBOL'),OrgDb = org.Hs.eg.db,drop = TRUE)
      enrichGO(gene = gdf$ENTREZID,OrgDb = org.Hs.eg.db,keyType = 'ENTREZID',ont = input$GOType,pvalueCutoff = input$GP,
               pAdjustMethod = input$GO_method,qvalueCutoff = input$GQ,minGSSize = 1,maxGSSize = 400,readable = TRUE)
    }
  })
  output$GOenrichment <- renderPlot({
    if(input$Difference_method == 'DESeq'){
      p <- barplot(sample_GO_data(),showCategory=input$GN,font.size=10)
      show(p)
    }
    if(input$Difference_method == 'DESeq2'){
      p <- barplot(group_GO_enrichment(),showCategory=input$GN,font.size=10)
      show(p)
    }
  })
  output$GObarplot <- renderPlot({
    if(input$Difference_method == 'DESeq'){
      p<- DOSE::dotplot(sample_GO_data(),showCategory=input$GN,font.size=10)
      show(p)
    }
    if(input$Difference_method == 'DESeq2'){
      p <- DOSE::dotplot(group_GO_enrichment(),showCategory=input$GN,font.size=10)
      show(p)
    }
  })
  output$GOgraph <- renderPlot({
    if(input$Difference_method == 'DESeq'){
      par(mar=c(5,5,6,10)+0.1,xpd=TRUE)
      plotGOgraph(sample_GO_data())
    }
    if(input$Difference_method == 'DESeq2'){
      par(mar=c(5,5,6,10)+0.1,xpd=TRUE)
      plotGOgraph(group_GO_enrichment())
    }
  })
}
app <-shinyApp(ui, server)
runApp(app,host = '0.0.0.0',port = 8080)
