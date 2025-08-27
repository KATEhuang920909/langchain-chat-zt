from PIL import Image, ImageDraw, ImageFont

def draw(logopath, infos, appname, apptype, viewpaths, resultpath):
    # 定义图片的尺寸和颜色
    banner_width = 800
    banner_height = 600
    banner_color = (255, 255, 255, 255)  # 白色背景

    # 创建一个空白的长方形图片
    banner_image = Image.new('RGBA', (banner_width, banner_height), banner_color)

    # 加载logo图片
    logo_image = Image.open(logopath)

    # 调整logo图片的大小（如果需要）
    logo_width, logo_height = 150, 150  # 假设logo的尺寸
    logo_image = logo_image.resize((logo_width, logo_height), Image.LANCZOS)

    # 计算logo在banner上的位置
    logo_position = (25, 25)

    # 将logo粘贴到banner上
    banner_image.paste(logo_image, logo_position)

    # 添加文字介绍
    text_lines = [
        "APP名称：" + appname,
        "APP类型：" + apptype,
        "开发者：" + infos['开发者'],
        "最新版本：" + infos['最新版本'],
        "应用分级" + infos['应用分级'],
    ]
    # 计算行间距
    line_spacing = 10

    font_path = 'C:/Windows/Fonts/simsun.ttc'  # 文件路径

    # 创建一个可以在图片上绘图的对象
    draw = ImageDraw.Draw(banner_image)

    # 使用系统字体
    font = ImageFont.truetype(font_path, 20)
    font2 = ImageFont.truetype(font_path, 20)

    # 设置文本颜色
    text_color = (0, 0, 0)  # 黑色文本

    # 计算并绘制每一行文本
    current_h = 20  # 起始高度
    for line in text_lines:
        text_width, text_height = draw.textsize(line, font=font)
        draw.text((220, current_h), line, fill=text_color, font=font)
        current_h += text_height + line_spacing

    # # 在图片上绘制文本
    draw.text((60,180), '图标参考', fill=text_color, font=font)
    draw.text((350,540), '界面图参考', fill=text_color, font=font)

    # 绘制预览图
    images = [Image.open(path) for path in viewpaths]

    # 定义每张图片的宽度和高度
    image_width = (banner_width-100) // 4  # 假设每张图片宽度为横幅宽度的1/4
    image_height = 300  # 假设每张图片的高度

    # 计算每张图片的位置
    positions = [(40 + i * (image_width + 10), 220) for i in range(4)]
    print(positions)
    # 调整每张图片的大小并粘贴到横幅上
    for img, pos in zip(images, positions):
        img = img.resize((image_width, image_height), Image.LANCZOS)
        banner_image.paste(img, pos)

    # 保存图片
    banner_image.save(resultpath, quality=95)

