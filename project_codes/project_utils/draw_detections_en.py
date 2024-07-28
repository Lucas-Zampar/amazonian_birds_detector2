import cv2

labels_translation = {
    'canario_do_amazonas':'of_yellow_finch', 
    'sanhaco_da_amazonia': 'blue_gray_tanager', 
    'sanhaco_do_coqueiro':'palm_tanager', 
    'chupim': 'shiny_cowbird', 
    'rolinha':'ground_dove', 
    'bem_te_vi':'great_kiskadee' 
}

color_per_label_hex = {
    'of_yellow_finch':'#fb8500', #orange
    'blue_gray_tanager': '#072ac8', #blue
    'palm_tanager':'#2b9348', #green
    'shiny_cowbird': '#9e4347', #brown
    'ground_dove':'#2d3752', #grey    
    'great_kiskadee':'#ffc600' #yellow
}

label_colors = {
    'canario_do_amazonas': '#fb8500', # laranja  #'#ffc600', #amarelo
    'chupim':'#9e4347', #marrom
    'rolinha':'#2d3752', #acizentado
    'sanhaco_da_amazonia':'#072ac8', #azul
    'sanhaco_do_coqueiro':'#2b9348', #verde
    'bem_te_vi': '#ffc600', #amarelo
}  

# define espessura da bounding box
bb_thickness = 4

# define parâmetros da fonte
font = cv2.FONT_HERSHEY_DUPLEX 
font_scale = 0.9
font_thickness = 1
line_type = cv2.LINE_AA

def hex_to_rgb(hex_string):
    '''
    Função responsável por converter o codificação da cor
    de hexadecimal para RGB
    '''
    return [int(hex_string[i:i+2], 16) for i in (1,3,5)]

# converte as cores de hexadecimal para RGB, uma vez que a OpenCV espera esse formato para colorir os retângulos das caixas
color_per_label_rgb = { key:hex_to_rgb(value) for key, value in color_per_label_hex.items()}

def draw_bb(frame, predictions):

    '''
    Função responsável por desenhar as caixas delimitadoras e 
    classes
    '''
    
    ## recupera predições
    bbox_list = predictions['detection']['bboxes']
    labels_list = predictions['detection']['labels']
    
    labels_list = [labels_translation[label] for label in labels_list]

    score_list = predictions['detection']['scores']
    detections = zip(bbox_list, labels_list, score_list)
    
    
    for bbox, label, score in detections:
        xmin, ymin, xmax, ymax = bbox.xyxy
        
        # define a cor da fonte
        if label != 'canario_do_amazonas':
            font_color = (255,255,255)
        else: 
            font_color = (0,0,0)
            
        
        ## canto superior esquerdo e canto inferir direito da bb
        bb_ltc = (xmin, ymin)
        bb_rbc = (xmax, ymax)
        
        text = f'{label} - {score:.2%}' 
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        
        ## canto inferior esquerdo do texto
        text_lbc = (xmin+10, ymin-10)
        
        ## canto superior esquerdo e canto inferior direito do bg 
        text_bg_ltc = (xmin, text_lbc[1]-text_h-10)
        text_bg_rbc = (text_lbc[0]+text_w,  ymin)
                
        color = color_per_label_rgb[label][::-1]
        frame = cv2.rectangle(frame, bb_ltc, bb_rbc, color, bb_thickness)    
        frame = cv2.rectangle(frame, text_bg_ltc, text_bg_rbc, color, -1)
        frame = cv2.putText(frame, text, text_lbc, font, font_scale, font_color, font_thickness, line_type)
    
    return frame






